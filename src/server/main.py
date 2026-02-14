# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better. 

import asyncio
import datetime
import json
import logging
import os
import re
import time
import traceback
import uuid
import base64
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from src.server.model_registry import ModelRegistry
from src.server.models.registration import ModelLoadConfig, ModelUnloadConfig
from src.server.models.openvino import OV_KokoroGenConfig
from src.server.models.ov_genai import OVGenAI_GenConfig, OVGenAI_WhisperGenConfig
from src.server.models.optimum import PreTrainedTokenizerConfig, RerankerConfig
from src.server.models.requests_internal import OpenArcBenchRequest
from src.server.models.requests_openai import (
    EmbeddingsRequest,
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
    OpenAIKokoroRequest,
    OpenAIWhisperRequest,
    RerankRequest,
)
from src.server.worker_registry import WorkerRegistry

logger = logging.getLogger(__name__)

#===============================================================#
# FastAPI configuration
#===============================================================#

# Initialize registries
_registry = ModelRegistry()
_workers = WorkerRegistry(_registry)

# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # Log incoming request
        logger.info(f"Request received: {request.method} {request.url.path} from {client_ip}")
        
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
        status_code=422,
        content={"status": "error", "detail": exc.errors()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    logger.error(f"Full traceback:\n{''.join(traceback.format_tb(exc.__traceback__))}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": str(exc)}
    )


#===============================================================#
# Tool calling helpers
#===============================================================#

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
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": str(data.get("name", "")),
                        "arguments": _format_tool_call_arguments(data.get("arguments", {})),
                    },
                })
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # Backward compatibility for plain JSON tool call outputs without tags.
    pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
    for match in re.findall(pattern, text, re.DOTALL):
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": str(data.get("name", "")),
                        "arguments": _format_tool_call_arguments(data.get("arguments", {})),
                    },
                })
        except json.JSONDecodeError:
            continue

    return tool_calls if tool_calls else None

#===============================================================#
# OpenArc internal
#===============================================================#

@app.post("/openarc/load", dependencies=[Depends(verify_api_key)])
async def load_model(load_config: ModelLoadConfig):
    try:
        model_id = await _registry.register_load(load_config)
        return {"model_id": model_id, "model_name": load_config.model_name, "status": "loaded"}
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
            raise HTTPException(status_code=404, detail=f"Model '{unload_config.model_name}' not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(exc)}")

@app.get("/openarc/status", dependencies=[Depends(verify_api_key)])
async def get_status():
    """Get registry status showing all loaded models."""
    return await _registry.status()

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
        
        logger.info(f"[bench] model={request.model} input_ids_len={len(request.input_ids)} metrics={metrics}")
        
        return {"metrics": metrics}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(exc)}")

#===============================================================#
# OpenAI-compatible endpoints
#===============================================================#

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def openai_list_models():
    """OpenAI-compatible endpoint that lists available models."""
    try:
        registry_status = await _registry.status()
        
        # Transform to OpenAI format
        models = []
        for model_name in registry_status["openai_model_names"]:
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(datetime.datetime.now().timestamp()),  # OpenAI uses Unix timestamp, we don't track this
                "owned_by": "OpenArc"
            })
        
        return {
            "object": "list",
            "data": models
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(exc)}")

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def openai_chat_completions(request: OpenAIChatCompletionRequest, raw_request: Request):
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
                    async for item in _workers.stream_generate(model_name, generation_config):
                        # Store request_id from gen_config for cancellation
                        if cancel_request_id is None and generation_config.request_id:
                            cancel_request_id = generation_config.request_id
                        
                        # Check for client disconnect
                        if await raw_request.is_disconnected():
                            if cancel_request_id:
                                await _workers.infer_cancel(cancel_request_id)
                                logger.info(f"[chat/completions] Client disconnected, cancelled {cancel_request_id}")
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
                                    'id': request_id,
                                    'object': 'chat.completion.chunk',
                                    'created': created_ts,
                                    'model': model_name,
                                    'choices': [{
                                        'index': 0,
                                        'delta': {
                                            'tool_calls': [{
                                                'index': idx,
                                                'id': tc['id'],
                                                'type': tc['type'],
                                                'function': {'name': tc['function']['name'], 'arguments': ''}
                                            }]
                                        },
                                        'finish_reason': None
                                    }]
                                }
                                yield (f"data: {json.dumps(tool_call_start)}\n\n").encode()
                                
                                # Stream arguments
                                tool_call_args = {
                                    'id': request_id,
                                    'object': 'chat.completion.chunk',
                                    'created': created_ts,
                                    'model': model_name,
                                    'choices': [{
                                        'index': 0,
                                        'delta': {
                                            'tool_calls': [{
                                                'index': idx,
                                                'function': {'arguments': tc['function']['arguments']}
                                            }]
                                        },
                                        'finish_reason': None
                                    }]
                                }
                                yield (f"data: {json.dumps(tool_call_args)}\n\n").encode()
                        elif not tool_calls and not tool_call_started:
                            # Regular content streaming
                            chunk_payload = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": item},
                                    "finish_reason": None,
                                }],
                            }
                            yield (f"data: {json.dumps(chunk_payload)}\n\n").encode()
                except asyncio.CancelledError:
                    if cancel_request_id:
                        await _workers.infer_cancel(cancel_request_id)
                        logger.info(f"[chat/completions] Task cancelled, cleaned up {cancel_request_id}")
                    raise
                
                # Final chunk
                prompt_tokens = (metrics_data or {}).get("input_token", 0)
                completion_tokens = (metrics_data or {}).get("new_token", 0)
                total_tokens = (metrics_data or {}).get("total_token", prompt_tokens + completion_tokens)
                
                finish_reason = "tool_calls" if tool_call_sent else "stop"
                
                final_payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
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
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        
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
                    async for item in _workers.stream_generate(model_name, generation_config):
                        # Store request_id from gen_config for cancellation
                        if cancel_request_id is None and generation_config.request_id:
                            cancel_request_id = generation_config.request_id
                        
                        # Check for client disconnect
                        if await raw_request.is_disconnected():
                            if cancel_request_id:
                                await _workers.infer_cancel(cancel_request_id)
                                logger.info(f"[completions] Client disconnected, cancelled {cancel_request_id}")
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
                        logger.info(f"[completions] Task cancelled, cleaned up {cancel_request_id}")
                    raise
                
                # Final stop signal per OpenAI SSE with usage
                prompt_tokens = (metrics_data or {}).get("input_token", 0)
                completion_tokens = (metrics_data or {}).get("new_token", 0)
                total_tokens = (metrics_data or {}).get("total_token", prompt_tokens + completion_tokens)
                
                logger.info(f"[completions] stream=true model={model_name} metrics={metrics_data}")
                
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
            
            logger.info(f"[completions] stream=false model={model_name} metrics={metrics}")

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
    language: Optional[str] = Form(None, description="Language of the input audio"),
    prompt: Optional[str] = Form(None, description="Optional text to guide the model"),
    response_format: Optional[str] = Form("json", description="Format of output"),
    temperature: Optional[float] = Form(0.0, description="Sampling temperature")
):
    try:
        logger.info(f'"{model}" request received')
        # Read the uploaded audio file
        audio_bytes = await file.read()

        # Convert to base64 for internal processing
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Create generation config with base64 audio
        gen_config = OVGenAI_WhisperGenConfig(audio_base64=audio_base64)

        # Process transcription
        result = await _workers.transcribe_whisper(model, gen_config)
        metrics = result.get("metrics", {})

        logger.info(f"[audio/transcriptions] model={model} metrics={metrics}")

        # Return based on response_format
        if response_format == "json":
            return {"text": result.get("text", "")}
        elif response_format == "verbose_json":
            return {
                "text": result.get("text", ""),
                "language": language,
                "duration": metrics.get("duration"),
                "metrics": metrics
            }
        else:  # text, srt, vtt
            return result.get("text", "")

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Transcription failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def openai_audio_speech(request: OpenAIKokoroRequest):
    """OpenAI-compatible endpoint for text-to-speech using Kokoro models.

    Returns a WAV file containing the synthesized speech.
    """
    try:
        logger.info(f'"{request.model}" request received')

        gen_config = OV_KokoroGenConfig(
            kokoro_message=request.input,
            voice=request.voice,
            lang_code=request.language,
            speed=request.speed,
            response_format=request.response_format
        )

        result = await _workers.generate_speech_kokoro(request.model, gen_config)
        metrics = result.get("metrics", {})
        
        logger.info(f"[audio/speech] model={request.model} voice={request.voice} metrics={metrics}")

        # Decode base64 audio and return as WAV file
        import base64
        audio_bytes = base64.b64decode(result.get("audio_base64", ""))

        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(exc)}")

@app.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
async def embeddings(request: EmbeddingsRequest):

    try:
        logger.info(f'"{request.model}" request received')

        tok_config = PreTrainedTokenizerConfig(
            text=request.input
        )

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
            embs.append({
                "index":i,
                "object":"embedding",
                "embedding":data[i]
            })

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
            docs.append({
                "index":i,
                "object":"ranked_documents",
                "ranked_documents":data[i]
            })

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
