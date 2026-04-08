# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better.

import asyncio
import datetime
import json
import logging
import os
import time
import traceback
import uuid
import base64
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional
import psutil
from pydantic import BaseModel

from src.server.downloader import global_downloader
from src.server.models.requests_management import (
    DownloaderRequest,
    DownloaderActionRequest,
)

from fastapi import Depends, FastAPI, HTTPException, Request, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from src.server.model_registry import ModelRegistry
from src.server.models.registration import ModelLoadConfig, ModelType, ModelUnloadConfig
from src.server.models.ov_genai import OVGenAI_GenConfig, OVGenAI_WhisperGenConfig
from src.server.models.optimum import PreTrainedTokenizerConfig, RerankerConfig
from src.server.models.requests_internal import OpenArcBenchRequest
from src.server.models.requests_openai import (
    EmbeddingsRequest,
    OpenArcASRConfig,
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
    OpenAISpeechRequest,
    RerankRequest,
)
from src.server.worker_registry import WorkerRegistry

logger = logging.getLogger(__name__)

# ===============================================================#
# FastAPI configuration
# ===============================================================#

# Initialize registries
_registry = ModelRegistry()
_workers = WorkerRegistry(_registry)


# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        # Log incoming request
        logger.info(
            f"Request received: {request.method} {request.url.path} from {client_ip}"
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} duration={process_time:.3f}s"
            )

            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"error={str(e)} duration={process_time:.3f}s"
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown"""
    # Startup: Load models from env var
    models = os.getenv("OPENARC_STARTUP_MODELS", "").strip()
    if models:
        from pathlib import Path

        config_file = Path(__file__).parent.parent.parent / "openarc_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

            for name in models.split(","):
                name = name.strip()
                model_config = config.get("models", {}).get(name)
                if not model_config:
                    logger.warning(f"Startup: model '{name}' not in config, skipping")
                    continue
                try:
                    await _registry.register_load(ModelLoadConfig(**model_config))
                    logger.info(f"Startup: loaded '{name}'")
                except Exception as e:
                    logger.error(f"Startup: failed to load '{name}': {e}")

    yield
    # Shutdown: (add cleanup here if needed)


app = FastAPI(lifespan=lifespan)

# API key authentication
API_KEY = os.getenv("OPENARC_API_KEY")
security = HTTPBearer()

# Add request logging middleware (before CORS so it logs all requests)
app.add_middleware(RequestLoggingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key provided in the Authorization header"""
    if credentials.credentials != API_KEY:
        logger.error(f"Invalid API key: {credentials.credentials}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}", exc_info=True)
    return JSONResponse(
        status_code=422, content={"status": "error", "detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    logger.error(f"Full traceback:\n{''.join(traceback.format_tb(exc.__traceback__))}")
    return JSONResponse(
        status_code=500, content={"status": "error", "detail": str(exc)}
    )


# ===============================================================#
# Tool calling helpers
# ===============================================================#


def _extract_hermes_tool_call_payloads(text: str) -> List[str]:
    open_tag = "<tool_call>"
    close_tag = "</tool_call>"
    payloads: List[str] = []
    cursor = 0

    while True:
        start = text.find(open_tag, cursor)
        if start < 0:
            break

        payload_start = start + len(open_tag)
        end = text.find(close_tag, payload_start)
        if end < 0:
            # Hermes parsers accept an open tool call until EOS.
            payload = text[payload_start:].strip()
            if payload:
                payloads.append(payload)
            break

        payload = text[payload_start:end].strip()
        if payload:
            payloads.append(payload)

        cursor = end + len(close_tag)

    return payloads


def _format_tool_call_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            return json.dumps(json.loads(arguments))
        except json.JSONDecodeError:
            return arguments
    return json.dumps(arguments)


def parse_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    tool_calls: List[Dict[str, Any]] = []

    # Hermes format: <tool_call>{...}</tool_call>, with EOS fallback for missing close tag.
    for payload in _extract_hermes_tool_call_payloads(text):
        try:
            data = json.loads(payload)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": str(data.get("name", "")),
                            "arguments": _format_tool_call_arguments(
                                data.get("arguments", {})
                            ),
                        },
                    }
                )
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    return None


# ===============================================================#
# OpenArc internal
# ===============================================================#


@app.post("/openarc/load", dependencies=[Depends(verify_api_key)])
async def load_model(load_config: ModelLoadConfig):
    try:
        model_id = await _registry.register_load(load_config)
        return {
            "model_id": model_id,
            "model_name": load_config.model_name,
            "status": "loaded",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(exc)}")


@app.post("/openarc/unload", dependencies=[Depends(verify_api_key)])
async def unload_model(unload_config: ModelUnloadConfig):
    """Unload a model by model_name."""
    try:
        success = await _registry.register_unload(unload_config.model_name)
        if success:
            return {"model_name": unload_config.model_name, "status": "unloading"}
        else:
            raise HTTPException(
                status_code=404, detail=f"Model '{unload_config.model_name}' not found"
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to unload model: {str(exc)}"
        )


@app.get("/openarc/status", dependencies=[Depends(verify_api_key)])
async def get_status():
    """Get registry status showing all loaded models."""
    return await _registry.status()


class UpdateModelConfigRequest(BaseModel):
    model_path: str
    config: Dict[str, Any]


@app.post("/openarc/models/update", dependencies=[Depends(verify_api_key)])
async def update_local_model_config(req: UpdateModelConfigRequest):
    import json
    from pathlib import Path

    target_path = Path(req.model_path)
    if not target_path.exists() or not target_path.is_dir():
        raise HTTPException(status_code=404, detail="Model directory not found")

    config_path = target_path / "openarc.json"

    current_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                current_config = json.load(f)
        except Exception:
            pass

    current_config.update(req.config)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(current_config, f, indent=4)
        return {"status": "success", "config": current_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@app.get("/openarc/models", dependencies=[Depends(verify_api_key)])
async def get_local_models(path: Optional[str] = None):
    import os
    import json
    from pathlib import Path

    if path:
        target_path = Path(path)
    else:
        target_path = Path.home() / ".cache" / "openarc" / "models"

    models = []
    if target_path.exists() and target_path.is_dir():
        for entry in target_path.iterdir():
            if entry.is_dir():
                folder_name = entry.name
                config_path = entry / "openarc.json"
                has_config = config_path.exists()

                model_name = folder_name
                model_type = None

                if has_config:
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            config_data = json.load(f)
                            model_name = config_data.get("model_name", model_name)
                            model_type = config_data.get("model_type")
                    except Exception:
                        pass
                else:
                    config_data = {}

                models.append(
                    {
                        "id": folder_name,
                        "path": str(entry),
                        "model_name": model_name,
                        "model_type": model_type,
                        "engine": config_data.get("engine"),
                        "vlm_type": config_data.get("vlm_type"),
                        "draft_model_path": config_data.get("draft_model_path"),
                        "draft_device": config_data.get("draft_device"),
                        "num_assistant_tokens": config_data.get("num_assistant_tokens"),
                        "assistant_confidence_threshold": config_data.get(
                            "assistant_confidence_threshold"
                        ),
                        "runtime_config": config_data.get("runtime_config", {}),
                        "has_config": has_config,
                    }
                )
    return {"models": models}


@app.get("/openarc/version", dependencies=[Depends(verify_api_key)])
async def get_version():
    return {"version": "v2.0.4"}


def get_hardware_metrics():
    gpus = []
    cpu_info = {"id": "CPU", "name": "System CPU"}
    npus = []

    # get cpu info natively using py-cpuinfo
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        cpu_info["name"] = info.get("brand_raw", "System CPU")
    except Exception as e:
        import logging

        logging.error(f"Failed to query CPU info: {e}")

    # get npu info via openvino (still needed since level zero is for gpu)
    try:
        import openvino as ov

        core = ov.Core()
        devices = core.available_devices
        for device in devices:
            if "NPU" in device:
                try:
                    name = core.get_property(device, "FULL_DEVICE_NAME")
                except Exception:
                    name = device
                npus.append({"id": device, "name": str(name)})
    except Exception as e:
        pass

    # get gpu metrics via the new custom extension
    try:
        import gpu_metrics

        data = gpu_metrics.get_gpu_metrics()
        for idx_str, gpu_data in data.items():
            name = gpu_data.get("name", f"Intel GPU {idx_str}")
            total_vram_mb = 0
            used_vram_mb = 0
            is_shared = False

            mem_list = gpu_data.get("memory", [])
            if mem_list and len(mem_list) > 0:
                total_vram_mb = mem_list[0].get("total", 0) // (1024 * 1024)
                used_vram_mb = mem_list[0].get("used", 0) // (1024 * 1024)
            else:
                import psutil

                vm = psutil.virtual_memory()
                total_vram_mb = vm.total // (1024 * 1024)
                is_shared = True

            gpus.append(
                {
                    "id": f"GPU.{idx_str}",
                    "name": str(name),
                    "total_vram": int(total_vram_mb),
                    "used_vram": int(used_vram_mb),
                    "usage": 0.0,  # usage is tough without polling, leaving 0
                    "is_shared": is_shared,
                }
            )
    except ImportError:
        import logging

        logging.warning(
            "gpu_metrics module not found. Intel GPU telemetry will be missing."
        )
    except Exception as e:
        import logging

        logging.error(f"Failed to fetch GPU metrics: {e}")

    return {"cpu": cpu_info, "gpus": gpus, "npus": npus}


@app.get("/openarc/metrics", dependencies=[Depends(verify_api_key)])
async def get_metrics():
    import psutil

    vm = psutil.virtual_memory()
    hw_metrics = get_hardware_metrics()

    return {
        "cpus": [
            {
                "id": hw_metrics["cpu"]["id"],
                "name": hw_metrics["cpu"]["name"],
                "cores": psutil.cpu_count(logical=False) or 1,
                "threads": psutil.cpu_count(logical=True) or 1,
                "usage": psutil.cpu_percent(),
            }
        ],
        "total_ram": vm.total // (1024 * 1024),
        "used_ram": vm.used // (1024 * 1024),
        "gpus": hw_metrics["gpus"],
        "npus": hw_metrics["npus"],
    }


@app.post("/openarc/downloader", dependencies=[Depends(verify_api_key)])
async def start_download(request: DownloaderRequest):
    success = await global_downloader.start(request.model_name, request.path)
    if success:
        return {"status": "success", "message": "Model download started successfully."}
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Download already in progress."},
    )


@app.get("/openarc/downloader", dependencies=[Depends(verify_api_key)])
async def list_downloads():
    return {"models": global_downloader.list_tasks()}


@app.delete("/openarc/downloader", dependencies=[Depends(verify_api_key)])
async def cancel_download(request: DownloaderActionRequest):
    if global_downloader.cancel(request.model_name):
        return {
            "status": "success",
            "message": "Model download cancelled successfully.",
        }
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Download task not found."},
    )


@app.post("/openarc/downloader/pause", dependencies=[Depends(verify_api_key)])
async def pause_download(request: DownloaderActionRequest):
    if global_downloader.pause(request.model_name):
        return {"status": "success", "message": "Model download paused successfully."}
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Active download task not found."},
    )


@app.post("/openarc/downloader/resume", dependencies=[Depends(verify_api_key)])
async def resume_download(request: DownloaderActionRequest):
    # resume is basicly start
    success = await global_downloader.start(request.model_name)
    if success:
        return {"status": "success", "message": "Model download resumed successfully."}
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Download already in progress."},
    )


@app.post("/openarc/bench", dependencies=[Depends(verify_api_key)])
async def benchmark(request: OpenArcBenchRequest):
    """Benchmark endpoint that accepts pre-encoded input_ids and returns only metrics."""
    try:
        config_kwargs = {
            "input_ids": request.input_ids,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "stream": False,  # Benchmarking is always non-streaming
        }
        # Remove keys with value None
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        generation_config = OVGenAI_GenConfig(**config_kwargs)

        result = await _workers.generate(request.model, generation_config)
        metrics = result.get("metrics", {}) or {}

        logger.info(
            f"[bench] model={request.model} input_ids_len={len(request.input_ids)} metrics={metrics}"
        )

        return {"metrics": metrics}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(exc)}")


# ===============================================================#
# OpenAI-compatible endpoints
# ===============================================================#


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def openai_list_models():
    """OpenAI-compatible endpoint that lists available models."""
    try:
        registry_status = await _registry.status()

        # Transform to OpenAI format
        models = []
        for model_name in registry_status["openai_model_names"]:
            models.append(
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(
                        datetime.datetime.now().timestamp()
                    ),  # OpenAI uses Unix timestamp, we don't track this
                    "owned_by": "OpenArc",
                }
            )

        return {"object": "list", "data": models}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list models: {str(exc)}"
        )


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest, raw_request: Request
):
    try:
        logger.info(f'"{request.model}" request received')

        config_kwargs = {
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.do_sample,
            "num_return_sequences": request.num_return_sequences,
            "stream": request.stream,
            "tools": request.tools,
        }
        # Remove keys with value None
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        generation_config = OVGenAI_GenConfig(**config_kwargs)

        model_name = request.model
        created_ts = int(time.time())
        request_id = f"ov-{uuid.uuid4().hex[:24]}"

        if generation_config.stream:

            async def event_stream() -> AsyncIterator[bytes]:
                # Stream OpenAI-compatible chunks
                accumulated_text = ""
                metrics_data = None
                tool_call_sent = False
                tool_call_started = False
                cancel_request_id = None

                try:
                    async for item in _workers.stream_generate(
                        model_name, generation_config
                    ):
                        # Store request_id from gen_config for cancellation
                        if cancel_request_id is None and generation_config.request_id:
                            cancel_request_id = generation_config.request_id

                        # Check for client disconnect
                        if await raw_request.is_disconnected():
                            if cancel_request_id:
                                await _workers.infer_cancel(cancel_request_id)
                                logger.info(
                                    f"[chat/completions] Client disconnected, cancelled {cancel_request_id}"
                                )
                            return

                        if isinstance(item, dict):
                            metrics_data = item.get("metrics", item)
                            continue

                        accumulated_text += item
                        if not tool_call_started:
                            tool_call_started = (
                                "<tool_call>" in accumulated_text
                                or "<tool_call" in accumulated_text
                                or "<tool_" in accumulated_text
                            )

                        tool_calls = parse_tool_calls(accumulated_text)

                        # If tool call detected and not yet sent, stream tool call deltas
                        if tool_calls and not tool_call_sent:
                            tool_call_sent = True
                            # Send tool call structure
                            for idx, tc in enumerate(tool_calls):
                                # Initial tool call with id, type, name
                                tool_call_start = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": idx,
                                                        "id": tc["id"],
                                                        "type": tc["type"],
                                                        "function": {
                                                            "name": tc["function"][
                                                                "name"
                                                            ],
                                                            "arguments": "",
                                                        },
                                                    }
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield (
                                    f"data: {json.dumps(tool_call_start)}\n\n"
                                ).encode()

                                # Stream arguments
                                tool_call_args = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": idx,
                                                        "function": {
                                                            "arguments": tc["function"][
                                                                "arguments"
                                                            ]
                                                        },
                                                    }
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield (
                                    f"data: {json.dumps(tool_call_args)}\n\n"
                                ).encode()
                        elif not tool_calls and not tool_call_started:
                            # Regular content streaming
                            chunk_payload = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": item},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield (f"data: {json.dumps(chunk_payload)}\n\n").encode()
                except asyncio.CancelledError:
                    if cancel_request_id:
                        await _workers.infer_cancel(cancel_request_id)
                        logger.info(
                            f"[chat/completions] Task cancelled, cleaned up {cancel_request_id}"
                        )
                    raise

                # Final chunk
                prompt_tokens = (metrics_data or {}).get("input_token", 0)
                completion_tokens = (metrics_data or {}).get("new_token", 0)
                total_tokens = (metrics_data or {}).get(
                    "total_token", prompt_tokens + completion_tokens
                )

                finish_reason = "tool_calls" if tool_call_sent else "stop"

                final_payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }
                yield (f"data: {json.dumps(final_payload)}\n\n").encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            result = await _workers.generate(model_name, generation_config)
            text = result.get("text", "")
            metrics = result.get("metrics", {}) or {}

            prompt_tokens = metrics.get("input_token", 0)
            completion_tokens = metrics.get("new_token", 0)
            total_tokens = metrics.get("total_token", prompt_tokens + completion_tokens)

            # Check for tool calls
            tool_calls = parse_tool_calls(text)
            message = {"role": "assistant"}
            finish_reason = "stop"

            if tool_calls:
                message["content"] = None
                message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
            else:
                message["content"] = text

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "metrics": metrics,  # OpenArc internal metrics
            }
            return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}")


@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
async def openai_completions(request: OpenAICompletionRequest, raw_request: Request):
    try:
        logger.info(f'"{request.model}" request received')
        prompt = (
            request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        )

        config_kwargs = {
            "prompt": prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.do_sample,
            "num_return_sequences": request.num_return_sequences,
            "stream": request.stream,
        }
        # Remove keys with value None
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        generation_config = OVGenAI_GenConfig(**config_kwargs)

        model_name = request.model
        created_ts = int(time.time())
        request_id = f"ov-{uuid.uuid4().hex[:24]}"

        if generation_config.stream:

            async def event_stream() -> AsyncIterator[bytes]:
                # Stream OpenAI-compatible chunks
                metrics_data = None
                cancel_request_id = None

                try:
                    async for item in _workers.stream_generate(
                        model_name, generation_config
                    ):
                        # Store request_id from gen_config for cancellation
                        if cancel_request_id is None and generation_config.request_id:
                            cancel_request_id = generation_config.request_id

                        # Check for client disconnect
                        if await raw_request.is_disconnected():
                            if cancel_request_id:
                                await _workers.infer_cancel(cancel_request_id)
                                logger.info(
                                    f"[completions] Client disconnected, cancelled {cancel_request_id}"
                                )
                            return

                        if isinstance(item, dict):
                            # Capture metrics for final usage payload
                            metrics_data = item.get("metrics", item)
                            continue
                        chunk_payload = {
                            "id": request_id,
                            "object": "text_completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": item,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield (f"data: {json.dumps(chunk_payload)}\n\n").encode()
                except asyncio.CancelledError:
                    if cancel_request_id:
                        await _workers.infer_cancel(cancel_request_id)
                        logger.info(
                            f"[completions] Task cancelled, cleaned up {cancel_request_id}"
                        )
                    raise

                # Final stop signal per OpenAI SSE with usage
                prompt_tokens = (metrics_data or {}).get("input_token", 0)
                completion_tokens = (metrics_data or {}).get("new_token", 0)
                total_tokens = (metrics_data or {}).get(
                    "total_token", prompt_tokens + completion_tokens
                )

                logger.info(
                    f"[completions] stream=true model={model_name} metrics={metrics_data}"
                )

                final_payload = {
                    "id": request_id,
                    "object": "text_completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }
                yield (f"data: {json.dumps(final_payload)}\n\n").encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            result = await _workers.generate(model_name, generation_config)
            text = result.get("text", "")
            metrics = result.get("metrics", {}) or {}

            prompt_tokens = metrics.get("input_token", 0)
            completion_tokens = metrics.get("new_token", 0)
            total_tokens = metrics.get("total_token", prompt_tokens + completion_tokens)

            logger.info(
                f"[completions] stream=false model={model_name} metrics={metrics}"
            )

            response = {
                "id": request_id,
                "object": "text_completion",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(exc)}")


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def openai_audio_transcriptions(
    file: UploadFile = File(..., description="The audio file to transcribe"),
    model: str = Form(..., description="ID of the model to use"),
    response_format: Optional[str] = Form("json", description="Format of output"),
    openarc_asr: Optional[str] = Form(
        None, description="JSON: OpenArcASRConfig with qwen3_asr params"
    ),
):
    try:
        logger.info(f'"{model}" request received')
        audio_bytes = await file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        selected_model_type = None
        async with _registry._lock:
            for record in _registry._models.values():
                if record.model_name == model:
                    selected_model_type = record.model_type
                    break

        if selected_model_type is None:
            raise ValueError(f"Model '{model}' is not loaded")

        normalized_model_type = ModelType(selected_model_type)

        if normalized_model_type == ModelType.QWEN3_ASR:
            if not openarc_asr:
                raise ValueError("openarc_asr required for Qwen3 ASR models")
            cfg = OpenArcASRConfig.model_validate(json.loads(openarc_asr))
            if not cfg.qwen3_asr:
                raise ValueError("openarc_asr.qwen3_asr required for Qwen3 ASR models")
            gen_config = cfg.qwen3_asr.model_copy(update={"audio_base64": audio_base64})
            result = await _workers.transcribe_qwen3_asr(model, gen_config)
        else:
            gen_config = OVGenAI_WhisperGenConfig(audio_base64=audio_base64)
            result = await _workers.transcribe_whisper(model, gen_config)
        metrics = result.get("metrics", {})

        logger.info(f"[audio/transcriptions] model={model} metrics={metrics}")

        # Return based on response_format
        if response_format == "json":
            return {"text": result.get("text", "")}
        elif response_format == "verbose_json":
            return {
                "text": result.get("text", ""),
                "language": metrics.get("language"),
                "duration": metrics.get("duration"),
                "metrics": metrics,
            }
        else:  # text, srt, vtt
            return result.get("text", "")

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Transcription failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def openai_audio_speech(request: OpenAISpeechRequest):
    """OpenAI-compatible endpoint for text-to-speech. Routes to Kokoro or Qwen3 TTS based on model type."""
    try:
        logger.info(f'"{request.model}" request received')

        selected_model_type = None
        async with _registry._lock:
            for record in _registry._models.values():
                if record.model_name == request.model:
                    selected_model_type = record.model_type
                    break

        if selected_model_type is None:
            raise ValueError(f"Model '{request.model}' is not loaded")

        normalized = ModelType(selected_model_type)

        if normalized in (
            ModelType.QWEN3_TTS_CUSTOM_VOICE,
            ModelType.QWEN3_TTS_VOICE_DESIGN,
            ModelType.QWEN3_TTS_VOICE_CLONE,
        ):
            if not request.openarc_tts or not request.openarc_tts.qwen3_tts:
                raise ValueError("openarc_tts.qwen3_tts required for Qwen3 TTS models")
            gen_config = request.openarc_tts.qwen3_tts
            gen_config.input = request.input
            if gen_config.stream:
                return StreamingResponse(
                    _workers.stream_generate_speech_qwen3_tts(
                        request.model, gen_config
                    ),
                    media_type="audio/L16;rate=24000;channels=1",
                )
            result = await _workers.generate_speech_qwen3_tts(request.model, gen_config)
        else:
            if not request.openarc_tts or not request.openarc_tts.kokoro:
                raise ValueError("openarc_tts.kokoro required for Kokoro models")
            gen_config = request.openarc_tts.kokoro
            gen_config.input = request.input
            result = await _workers.generate_speech_kokoro(request.model, gen_config)

        metrics = result.get("metrics", {})
        logger.info(
            f"[audio/speech] model={request.model} voice={request.voice} metrics={metrics}"
        )

        audio_bytes = base64.b64decode(result.get("audio_base64", ""))
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Speech synthesis failed: {str(exc)}"
        )


@app.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
async def embeddings(request: EmbeddingsRequest):

    try:
        logger.info(f'"{request.model}" request received')

        tok_config = PreTrainedTokenizerConfig(text=request.input)

        if request.config:
            tok_config = request.config
            if not tok_config.text:
                tok_config.text = request.input

        if not tok_config.max_length and request.dimensions:
            tok_config.max_length = request.dimensions

        model_name = request.model
        created_ts = int(time.time())
        request_id = f"ov-{uuid.uuid4().hex[:24]}"

        result = await _workers.embed(model_name, tok_config)
        data = result.get("data", None)
        metrics = result.get("metrics", {}) or {}

        prompt_tokens = metrics.get("input_token", 0)
        total_tokens = metrics.get("total_token", prompt_tokens)

        logger.info(f"[embeddings] model={model_name} metrics={metrics}")

        embs = []
        for i in range(len(data)):
            embs.append({"index": i, "object": "embedding", "embedding": data[i]})

        response = {
            "id": request_id,
            "object": "list",
            "created": created_ts,
            "model": model_name,
            "data": embs,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            },
        }

        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(exc)}")


@app.post("/v1/rerank", dependencies=[Depends(verify_api_key)])
async def rerank(request: RerankRequest):

    try:
        logger.info(f'"{request.model}" request received')
        config_data = {"query": request.query, "documents": request.documents}
        if request.prefix is not None:
            config_data["prefix"] = request.prefix
        if request.suffix is not None:
            config_data["suffix"] = request.suffix
        if request.instruction is not None:
            config_data["instruction"] = request.instruction

        rr_config = RerankerConfig.model_validate(config_data)

        model_name = request.model
        created_ts = int(time.time())
        request_id = f"ov-{uuid.uuid4().hex[:24]}"

        result = await _workers.rerank(model_name, rr_config)
        data = result.get("data", None)
        metrics = result.get("metrics", {}) or {}

        prompt_tokens = metrics.get("input_token", 0)
        total_tokens = metrics.get("total_token", prompt_tokens)

        docs = []
        for i in range(len(data)):
            docs.append(
                {"index": i, "object": "ranked_documents", "ranked_documents": data[i]}
            )

        response = {
            "id": request_id,
            "object": "list",
            "created": created_ts,
            "model": model_name,
            "data": docs,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            },
        }

        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(exc)}")
