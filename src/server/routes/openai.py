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
from src.server.schemas.modeling.contract_kokoro import (
    KokoroLanguage,
    KokoroVoice,
    OV_KokoroGenConfig,
)
from src.server.schemas.modeling.contract_optimum_emb import PreTrainedTokenizerConfig
from src.server.schemas.modeling.contract_optimum_rerank import RerankerConfig
from src.server.schemas.modeling.contract_qwen3asr import OV_Qwen3ASRGenConfig
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.modeling.contract_qwen3tts import (
    OV_Qwen3TTSCustomVoice,
    OV_Qwen3TTSVoiceClone,
    OV_Qwen3TTSVoiceDesign,
)
from src.server.schemas.modeling.contract_whisper import OVGenAI_WhisperGenConfig
from src.server.schemas.registration import ModelLoadConfig, ModelType, ModelUnloadConfig
from src.server.schemas.requests_internal import OpenArcBenchRequest
from src.server.schemas.requests_openai import (
    EmbeddingsRequest,
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
    OpenAISpeechRequest,
    OpenArcASRConfig,
    RerankRequest,
)
from src.server.utils.merge import build_config, defaults_for_record
from src.engine.ov_genai.tool_parse import gemma4, hermes, museglimmer, qwen35

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


# Tool-call parser modules keyed by ModelLoadConfig.tool_call_parser value.
_TOOL_PARSERS = {
    "qwen35": qwen35,
    "hermes": hermes,
    "gemma4": gemma4,
    "museglimmer": museglimmer,
}


def _get_record(model_name: str):
    """Return the loaded ModelRecord for a model_name, or None.

    Caller must not hold _registry._lock (this takes it). The returned record's
    plain-dict config blocks are safe to read outside the lock.
    """
    for record in list(_registry._models.values()):
        if record.model_name == model_name:
            return record
    return None


def _record_defaults(model_name: str) -> Dict[str, Any]:
    """Resolve the config.yaml defaults that apply to a loaded model."""
    record = _get_record(model_name)
    if record is None:
        return {}
    return defaults_for_record(record)


def _prepend_system_instruction(messages: Any, instruction: str) -> Any:
    if not isinstance(messages, list):
        return messages
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        messages[0]["content"] = f"{content}\n\n{instruction}".strip()
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages


def _apply_tool_choice(
    messages: Any,
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[Any],
    parallel_tool_calls: Optional[bool],
) -> tuple[Any, Optional[List[Dict[str, Any]]]]:
    """Translate OpenAI tool-choice controls into Qwen prompt constraints."""
    if tool_choice == "none":
        return messages, None

    effective_tools = tools
    instruction = ""

    if tool_choice in (None, "auto"):
        pass
    elif tool_choice == "required":
        if not tools:
            raise ValueError("tool_choice='required' needs at least one tool")
        instruction = (
            "You must emit at least one tool call now. Do not answer in natural "
            "language. Select the best available tool and infer reasonable arguments."
        )
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function") or {}
        name = function.get("name")
        if tool_choice.get("type") != "function" or not name:
            raise ValueError("Named tool_choice must specify type='function' and function.name")
        effective_tools = [
            tool for tool in tools or []
            if (tool.get("function") or {}).get("name") == name
        ]
        if not effective_tools:
            raise ValueError(f"tool_choice references unknown function '{name}'")
        instruction = (
            f"You must call the provided '{name}' tool now. Output exactly one tool "
            "call and no natural-language answer. Infer reasonable arguments from "
            "the user's request."
        )
    else:
        raise ValueError("tool_choice must be 'auto', 'none', 'required', or a named function")

    if parallel_tool_calls is False and effective_tools:
        suffix = "Call at most one tool in your response."
        instruction = f"{instruction} {suffix}".strip()

    if instruction:
        messages = _prepend_system_instruction(messages, instruction)
    return messages, effective_tools


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

        tool_parser_name = None
        async with _registry._lock:
            for record in _registry._models.values():
                if record.model_name == request.model:
                    # record.tool_call_parser is a ToolCallParser enum; the
                    # parser registry below is keyed by its string value.
                    parser_enum = record.tool_call_parser
                    tool_parser_name = parser_enum.value if parser_enum else None
                    break

        if tool_parser_name is None and request.tools:
            raise ValueError(
                f"Model '{request.model}' has no tool_call_parser configured; "
                "set one in the model config (e.g. 'openarc add --tool-call-parser qwen35|hermes|gemma4|museglimmer')"
            )
        parser_module = _TOOL_PARSERS.get(tool_parser_name) if tool_parser_name else None

        messages, tools = _apply_tool_choice(
            request.messages,
            request.tools,
            request.tool_choice,
            request.parallel_tool_calls,
        )
        chat_template_kwargs = dict(request.chat_template_kwargs or {})
        if request.tool_choice == "required" or isinstance(request.tool_choice, dict):
            chat_template_kwargs.setdefault("enable_thinking", False)

        config_kwargs = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.do_sample,
            "num_return_sequences": request.num_return_sequences,
            "seed": request.seed,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }
        if parser_module is not None:
            config_kwargs["tool_call_parser"] = tool_parser_name

        # Layer the model's config.yaml sampler defaults under the request.
        # Precedence: request-time > config.yaml > engine default.
        generation_config = build_config(
            OVGenAI_GenConfig,
            request={**config_kwargs, "chat_template_kwargs": chat_template_kwargs},
            defaults=_record_defaults(request.model),
            messages=messages,
            tools=tools,
            stream=request.stream,
        )

        model_name = request.model
        created_ts = int(time.time())
        request_id = f"ov-{uuid.uuid4().hex[:24]}"

        thinking_enabled = True
        if chat_template_kwargs:
            thinking_enabled = chat_template_kwargs.get(
                "enable_thinking", True
            )

        if generation_config.stream:

            async def event_stream() -> AsyncIterator[bytes]:
                metrics_data = None
                tool_call_sent = False
                cancel_request_id = None
                stream_parser = None
                # qwen35 tool requests and gemma4/museglimmer requests (all
                # museglimmer traffic; gemma4 when tools or thinking) stream
                # through the engine's tool streamers (parsed deltas on the
                # worker queue); the text-delta facade only handles no-tool
                # qwen35 requests.
                engine_tool_stream = (
                    parser_module is qwen35 and bool(tools)
                ) or (
                    parser_module is gemma4
                    and gemma4.wants_engine_stream(tools, thinking_enabled)
                ) or (
                    parser_module is museglimmer
                    and museglimmer.wants_engine_stream(tools, thinking_enabled)
                )
                if parser_module is qwen35 and not engine_tool_stream:
                    stream_parser = qwen35.Qwen35StreamParser(
                        tools, enable_thinking=thinking_enabled
                    )
                elif parser_module is hermes:
                    stream_parser = hermes.HermesStreamParser(
                        enable_thinking=thinking_enabled
                    )

                def _chunk(delta: dict) -> bytes:
                    return (
                        f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': model_name, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
                    ).encode()

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
                            if item.get("error"):
                                raise RuntimeError(item["error"])
                            if "chat_delta" in item:
                                # Parsed deltas from Qwen35ToolCallStreamer
                                for delta in item["chat_delta"]:
                                    if "tool_calls" in delta:
                                        tool_call_sent = True
                                    yield _chunk(delta)
                                continue
                            metrics_data = item.get("metrics", item)
                            continue

                        if stream_parser is not None:
                            for delta in stream_parser.feed(item):
                                if "tool_calls" in delta:
                                    tool_call_sent = True
                                yield _chunk(delta)
                        else:
                            yield _chunk({"content": item})
                except asyncio.CancelledError:
                    if cancel_request_id:
                        await _workers.infer_cancel(cancel_request_id)
                        logger.info(
                            f"[chat/completions] Task cancelled, cleaned up {cancel_request_id}"
                        )
                    raise

                if stream_parser is not None:
                    for delta in stream_parser.finish():
                        if "tool_calls" in delta:
                            tool_call_sent = True
                        yield _chunk(delta)

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

            if parser_module is not None:
                reasoning_text, content_text, tool_calls = parser_module.parse_generation(
                    text, tools, thinking_enabled
                )
            else:
                reasoning_text, content_text, tool_calls = None, text, None
            message = {"role": "assistant"}
            finish_reason = "stop"

            if reasoning_text:
                message["reasoning_content"] = reasoning_text

            if tool_calls:
                message["content"] = content_text or None
                message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
            else:
                message["content"] = content_text if content_text else text

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
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.do_sample,
            "num_return_sequences": request.num_return_sequences,
        }

        # Layer the model's config.yaml sampler defaults under the request.
        generation_config = build_config(
            OVGenAI_GenConfig,
            request=config_kwargs,
            defaults=_record_defaults(request.model),
            prompt=prompt,
            stream=request.stream,
        )

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
                            if item.get("error"):
                                raise RuntimeError(item["error"])
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
    language: Optional[str] = Form(
        None,
        description=(
            "Language of the input audio as an ISO-639-1 code (e.g. 'en') or "
            "full name (e.g. 'English'). When set, overrides "
            "openarc_asr.qwen3_asr.language; when unset, that value is used."
        ),
    ),
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
            payload = json.loads(openarc_asr) if openarc_asr else {}
            if not payload.get("qwen3_asr"):
                # Fall back to defaults if qwen3_asr config is not provided
                payload["qwen3_asr"] = {}

            cfg = OpenArcASRConfig.model_validate(payload)
            # Layer the model's config.yaml qwen3_asr_config under the request.
            # The original handler used model_copy(update=...), which skips
            # validation; passing through build_config re-validates, so only
            # real values (not unset FastAPI Form sentinels) may be injected.
            request_fields = cfg.qwen3_asr.model_dump(exclude_unset=True)
            request_fields["audio_base64"] = audio_base64
            if isinstance(language, str) and language:
                # Whisper-style top-level `language` takes precedence; otherwise
                # fall back to openarc_asr.qwen3_asr.language (current behavior).
                request_fields["language"] = language
            gen_config = build_config(
                OV_Qwen3ASRGenConfig,
                request=request_fields,
                defaults=_record_defaults(model),
            )
            result = await _workers.transcribe_qwen3_asr(model, gen_config)
        else:
            gen_config = OVGenAI_WhisperGenConfig(audio_base64=audio_base64)
            result = await _workers.transcribe_whisper(model, gen_config)

        metrics: Dict[str, Any] = result.get("metrics", {})
        logger.info(f"[audio/transcriptions] model={model} metrics={metrics}")

        if response_format == "json":
            return {"text": result.get("text", "")}
        elif response_format == "verbose_json":
            return {
                "text": result.get("text", ""),
                "language": metrics.get("language"),
                "duration": metrics.get("duration") or metrics.get("audio_duration_sec"),
                "segments": result.get("segments", []),
                "metrics": metrics,
            }
        elif response_format == "diarized_json":
            return {
                "duration": metrics.get("duration") or metrics.get("audio_duration_sec"),
                "segments": result.get("segments", []),
                "task": "transcribe",
                "text": result.get("text", ""),
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
        tts_defaults = _record_defaults(request.model)

        if normalized in (
            ModelType.QWEN3_TTS_CUSTOM_VOICE,
            ModelType.QWEN3_TTS_VOICE_DESIGN,
            ModelType.QWEN3_TTS_VOICE_CLONE,
        ):
            if not request.openarc_tts:
                raise ValueError("openarc_tts required for Qwen3 TTS models")
            _qwen3_tts_field = {
                ModelType.QWEN3_TTS_CUSTOM_VOICE: "qwen3_tts_custom_voice",
                ModelType.QWEN3_TTS_VOICE_DESIGN: "qwen3_tts_voice_design",
                ModelType.QWEN3_TTS_VOICE_CLONE: "qwen3_tts_voice_clone",
            }[normalized]
            _contract = {
                ModelType.QWEN3_TTS_CUSTOM_VOICE: OV_Qwen3TTSCustomVoice,
                ModelType.QWEN3_TTS_VOICE_DESIGN: OV_Qwen3TTSVoiceDesign,
                ModelType.QWEN3_TTS_VOICE_CLONE: OV_Qwen3TTSVoiceClone,
            }[normalized]
            supplied = getattr(request.openarc_tts, _qwen3_tts_field)
            if supplied is None:
                raise ValueError(f"openarc_tts.{_qwen3_tts_field} required for {normalized.value} models")
            # Seed the contract from config.yaml, letting explicitly supplied
            # request fields win. Anything neither layer sets stays unset, so
            # the guards below still see "not provided by the caller".
            gen_config = build_config(
                _contract,
                request=supplied.model_dump(exclude_unset=True),
                defaults=tts_defaults,
                input=request.input,
            )
            if request.language is not None and "language" not in gen_config.model_fields_set:
                gen_config.language = request.language
            if (
                request.instructions is not None
                and hasattr(gen_config, "instruct")
                and "instruct" not in gen_config.model_fields_set
            ):
                gen_config.instruct = request.instructions
            if (
                request.voice is not None
                and isinstance(gen_config, OV_Qwen3TTSCustomVoice)
                and "speaker" not in gen_config.model_fields_set
            ):
                gen_config.speaker = request.voice
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
            gen_config = build_config(
                OV_KokoroGenConfig,
                request=request.openarc_tts.kokoro.model_dump(exclude_unset=True),
                defaults=tts_defaults,
                input=request.input,
            )
            if request.voice is not None and "voice" not in gen_config.model_fields_set:
                try:
                    gen_config.voice = KokoroVoice(request.voice)
                except ValueError:
                    raise ValueError(f"Unknown Kokoro voice: '{request.voice}'. See KokoroVoice for valid values.")
            if request.language is not None and "lang_code" not in gen_config.model_fields_set:
                try:
                    gen_config.lang_code = KokoroLanguage(request.language)
                except ValueError:
                    raise ValueError(f"Unknown Kokoro language code: '{request.language}'. See KokoroLanguage for valid values.")
            if "response_format" not in gen_config.model_fields_set and request.response_format is not None:
                gen_config.response_format = request.response_format
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
