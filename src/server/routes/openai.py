import asyncio
import base64
import datetime
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.server.deps import _registry, _workers, verify_api_key
from src.server.models.optimum import PreTrainedTokenizerConfig, RerankerConfig
from src.server.models.ov_genai import OVGenAI_GenConfig, OVGenAI_WhisperGenConfig
from src.server.models.registration import ModelType
from src.server.models.requests_openai import (
    EmbeddingsRequest,
    OpenArcASRConfig,
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
    OpenAISpeechRequest,
    RerankRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


# ---- tool call helpers ----

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
            # hermes parsers accept an open tool call until EOS
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

    return tool_calls if tool_calls else None


# ---- endpoints ----

@router.get("/models", dependencies=[Depends(verify_api_key)])
async def openai_list_models():
    try:
        registry_status = await _registry.status()

        models = []
        for model_name in registry_status["openai_model_names"]:
            models.append(
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(datetime.datetime.now().timestamp()),
                    "owned_by": "OpenArc",
                }
            )

        return {"object": "list", "data": models}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list models: {str(exc)}"
        )


@router.post("/chat/completions", dependencies=[Depends(verify_api_key)])
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
            "seed": request.seed,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        generation_config = OVGenAI_GenConfig(**config_kwargs)

        model_name = request.model
        created_ts = int(time.time())
        request_id = f"ov-{uuid.uuid4().hex[:24]}"

        if generation_config.stream:

            async def event_stream() -> AsyncIterator[bytes]:
                accumulated_text = ""
                metrics_data = None
                tool_call_sent = False
                tool_call_started = False
                cancel_request_id = None

                try:
                    async for item in _workers.stream_generate(
                        model_name, generation_config
                    ):
                        if cancel_request_id is None and generation_config.request_id:
                            cancel_request_id = generation_config.request_id

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

                        if tool_calls and not tool_call_sent:
                            tool_call_sent = True
                            for idx, tc in enumerate(tool_calls):
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
                                                            "name": tc["function"]["name"],
                                                            "arguments": "",
                                                        },
                                                    }
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield (f"data: {json.dumps(tool_call_start)}\n\n").encode()

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
                                                            "arguments": tc["function"]["arguments"]
                                                        },
                                                    }
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield (f"data: {json.dumps(tool_call_args)}\n\n").encode()
                        elif not tool_calls and not tool_call_started:
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

            tool_calls = parse_tool_calls(text)
            message = {"role": "assistant"}
            finish_reason = "stop"

            if tool_calls:
                message["content"] = None
                message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
            else:
                message["content"] = text

            return {
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
                "metrics": metrics,
            }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}")


@router.post("/completions", dependencies=[Depends(verify_api_key)])
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
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        generation_config = OVGenAI_GenConfig(**config_kwargs)

        model_name = request.model
        created_ts = int(time.time())
        request_id = f"ov-{uuid.uuid4().hex[:24]}"

        if generation_config.stream:

            async def event_stream() -> AsyncIterator[bytes]:
                metrics_data = None
                cancel_request_id = None

                try:
                    async for item in _workers.stream_generate(
                        model_name, generation_config
                    ):
                        if cancel_request_id is None and generation_config.request_id:
                            cancel_request_id = generation_config.request_id

                        if await raw_request.is_disconnected():
                            if cancel_request_id:
                                await _workers.infer_cancel(cancel_request_id)
                                logger.info(
                                    f"[completions] Client disconnected, cancelled {cancel_request_id}"
                                )
                            return

                        if isinstance(item, dict):
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

            return {
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(exc)}")


@router.post("/audio/transcriptions", dependencies=[Depends(verify_api_key)])
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

        if response_format == "json":
            return {"text": result.get("text", "")}
        elif response_format == "verbose_json":
            return {
                "text": result.get("text", ""),
                "language": metrics.get("language"),
                "duration": metrics.get("duration"),
                "metrics": metrics,
            }
        else:
            return result.get("text", "")

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Transcription failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")


@router.post("/audio/speech", dependencies=[Depends(verify_api_key)])
async def openai_audio_speech(request: OpenAISpeechRequest):
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
                    _workers.stream_generate_speech_qwen3_tts(request.model, gen_config),
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


@router.post("/embeddings", dependencies=[Depends(verify_api_key)])
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

        embs = [{"index": i, "object": "embedding", "embedding": data[i]} for i in range(len(data))]

        return {
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(exc)}")


@router.post("/rerank", dependencies=[Depends(verify_api_key)])
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

        docs = [
            {"index": i, "object": "ranked_documents", "ranked_documents": data[i]}
            for i in range(len(data))
        ]

        return {
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(exc)}")
