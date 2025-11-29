# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better. 

import asyncio
import base64
import datetime
import json
import logging
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from PIL import Image

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pydantic import BaseModel

from src.server.model_registry import ModelLoadConfig, ModelRegistry, ModelUnloadConfig
from src.server.models.openvino import OV_KokoroGenConfig
from src.server.models.ov_genai import OVGenAI_GenConfig, OVGenAI_WhisperGenConfig
from src.server.models.optimum import PreTrainedTokenizerConfig, RerankerConfig
from src.server.worker_registry import WorkerRegistry

logger = logging.getLogger(__name__)

#===============================================================#
# FastAPI configuration
#===============================================================#

# Initialize registries
_registry = ModelRegistry()
_workers = WorkerRegistry(_registry)

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

def parse_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    # Find all potential JSON objects
    pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return None
    
    tool_calls = []
    for match in matches:
        try:
            data = json.loads(match)
            # Check if it has the expected structure
            if "name" in data and "arguments" in data:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": data.get("name", ""),
                        "arguments": json.dumps(data.get("arguments", {}))
                    }
                })
        except json.JSONDecodeError:
            pass
    
    return tool_calls if tool_calls else None

#===============================================================#
# Utility Functions for Template Application and Tokenization
#===============================================================#

def _load_openarc_config() -> dict:
    """Load openarc_config.json from project root."""
    config_file = Path(__file__).parent.parent.parent / "openarc_config.json"
    if not config_file.exists():
        raise ValueError(f"Config not found: {config_file}")

    with open(config_file) as f:
        return json.load(f)


def _process_vlm_messages(
    messages: List[Dict[str, Any]],
    vlm_type: str
) -> List[Dict[str, Any]]:
    """
    Process multimodal messages for VLM models, extracting images and inserting vision tokens.

    Args:
        messages: OpenAI-format messages with potential image content
        vlm_type: Vision token type (e.g., "qwen2vl", "llava15")

    Returns:
        Processed messages with vision tokens inserted

    Raises:
        ValueError: If unknown VLM type or invalid image format
    """
    from src.server.models.ov_genai import VLM_VISION_TOKENS

    vision_token = VLM_VISION_TOKENS.get(vlm_type)
    if not vision_token:
        raise ValueError(f"Unknown VLM type: {vlm_type}. Supported: {list(VLM_VISION_TOKENS.keys())}")

    processed_messages = []
    image_count = 0

    for message in messages:
        content = message.get("content", "")

        # Handle multimodal content (list format)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        # Extract base64 image (we don't actually process it for template application)
                        image_url = item.get("image_url", {})
                        if isinstance(image_url, dict) and isinstance(image_url.get("url", ""), str):
                            url = image_url["url"]
                            if url.startswith("data:image/"):
                                # Valid image data URL found
                                image_count += 1

                                # Insert vision token
                                token_str = vision_token.replace("{i}", str(image_count - 1)) if "{i}" in vision_token else vision_token
                                text_parts.append(f"{token_str}")
                            else:
                                logger.warning(f"[VLM] Non-data URL images not supported: {url[:50]}...")

                    elif item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    else:
                        logger.warning(f"[VLM] Unknown content type: {item.get('type')}")

            # Reconstruct message with text only
            new_message = message.copy()
            new_message["content"] = " ".join(text_parts)
            processed_messages.append(new_message)
        else:
            # Text-only message
            processed_messages.append(message)

    logger.info(f"[VLM] Processed {len(messages)} messages, found {image_count} images")
    return processed_messages


async def load_tokenizer_for_model(model_path: str):
    """
    Load tokenizer from model path (async, cached by HuggingFace).

    Args:
        model_path: Path to model directory containing tokenizer files

    Returns:
        AutoTokenizer instance

    Raises:
        ValueError: If model path doesn't exist
    """
    from transformers import AutoTokenizer

    if not Path(model_path).exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    return await asyncio.to_thread(AutoTokenizer.from_pretrained, model_path)


async def apply_chat_template_for_model(
    model_path: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    vlm_type: Optional[str] = None
) -> str:
    """
    Apply chat template to messages using the model's tokenizer.

    Args:
        model_path: Path to model directory
        messages: OpenAI-format chat messages
        tools: Optional tool definitions for function calling
        add_generation_prompt: Whether to add generation prompt tag (e.g., <|assistant|>)
        chat_template_kwargs: Additional parameters passed to template system
        vlm_type: Optional VLM type for multimodal support (e.g., "qwen2vl", "gemma3")

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If model path is invalid or tokenizer can't be loaded
        RuntimeError: If chat template application fails
    """
    try:
        # Process VLM messages if this is a multimodal model
        if vlm_type:
            messages = _process_vlm_messages(messages, vlm_type)

        tokenizer = await load_tokenizer_for_model(model_path)

        kwargs = chat_template_kwargs or {}
        return tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,  # Return string, not token IDs
            **kwargs
        )
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Failed to apply chat template: {str(e)}")


async def tokenize_content(
    model_path: str,
    content: str,
    add_special: bool = True
) -> List[int]:
    """
    Tokenize text content and return token IDs.

    Args:
        model_path: Path to model directory
        content: Text content to tokenize
        add_special: Whether to add special tokens

    Returns:
        List of token IDs

    Raises:
        ValueError: If model path is invalid or tokenizer can't be loaded
    """
    tokenizer = await load_tokenizer_for_model(model_path)
    return tokenizer.encode(content, add_special_tokens=add_special)

#===============================================================#
# Request Models
#===============================================================#

class OpenArcBenchRequest(BaseModel):
    model: str
    input_ids: List[int]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

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

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: Any
    tools: Optional[List[Dict[str, Any]]] = None
    stream: Optional[bool] = None
    
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    num_return_sequences: Optional[int] = None

class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stream: Optional[bool] = None
    
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    num_return_sequences: Optional[int] = None

class OpenAIWhisperRequest(BaseModel):
    model: str
    audio_base64: str

class OpenAIKokoroRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    speed: Optional[float] = None
    language: Optional[str] = None
    response_format: Optional[str] = "wav"

# https://platform.openai.com/docs/api-reference/embeddings
class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str], List[List[str]]]
    dimensions: Optional[int] = None
    encoding_format: Optional[str] = "float" #not implemented
    user: Optional[str] = None, #not implemented
    #end of openai api
    config: Optional[PreTrainedTokenizerConfig] = None

# No openai api to reference
class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    prefix:Optional[str] = None
    suffix:Optional[str] = None
    instruction:Optional[str] = None

# Template application and tokenization endpoints (llama.cpp compatible)
class ApplyTemplateRequest(BaseModel):
    """Request model for /apply-template endpoint."""
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    add_generation_prompt: Optional[bool] = True
    chat_template_kwargs: Optional[Dict[str, Any]] = {}

class ApplyTemplateResponse(BaseModel):
    """Response model for /apply-template endpoint."""
    prompt: str

class TokenizeRequest(BaseModel):
    """Request model for /tokenize endpoint."""
    model: Optional[str] = None
    content: str
    add_special: Optional[bool] = True

class TokenizeResponse(BaseModel):
    """Response model for /tokenize endpoint."""
    tokens: List[int]

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
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    try:
    
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
                
                async for item in _workers.stream_generate(model_name, generation_config):
                    if isinstance(item, dict):
                        metrics_data = item.get("metrics", item)
                        continue
                    
                    accumulated_text += item
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
                    elif not tool_calls:
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
async def openai_completions(request: OpenAICompletionRequest):
    try:
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
                async for item in _workers.stream_generate(model_name, generation_config):
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
async def openai_audio_transcriptions(request: OpenAIWhisperRequest):
    try:
        gen_config = OVGenAI_WhisperGenConfig(audio_base64=request.audio_base64)
        result = await _workers.transcribe_whisper(request.model, gen_config)
        metrics = result.get("metrics", {})
        
        logger.info(f"[audio/transcriptions] model={request.model} metrics={metrics}")
        
        return {"text": result.get("text", ""), "metrics": metrics}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")

@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def openai_audio_speech(request: OpenAIKokoroRequest):
    """OpenAI-compatible endpoint for text-to-speech using Kokoro models.

    Returns a WAV file containing the synthesized speech.
    """
    try:

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

#===============================================================#
# Template Application and Tokenization Endpoints
#===============================================================#

@app.post("/apply-template", dependencies=[Depends(verify_api_key)])
async def apply_template(request: ApplyTemplateRequest):
    """
    Apply chat template to messages without performing inference.

    This endpoint formats OpenAI-compatible chat messages according to the model's
    chat template, returning only the formatted prompt string. It matches the
    llama.cpp /apply-template specification.

    The endpoint works independently of model loading - it loads only the tokenizer
    on-demand, making it lightweight and fast.

    Returns:
        {"prompt": "formatted prompt string"}

    Raises:
        400: Invalid request (missing messages, bad model path, etc.)
        500: Template application failed
    """
    try:
        # Load config to get model_path
        config = _load_openarc_config()

        # If model not specified, use first available model
        if not request.model:
            models = list(config.get("models", {}).keys())
            if not models:
                raise HTTPException(status_code=400, detail="No models configured")
            request.model = models[0]
            logger.info(f"[apply-template] No model specified, using first available: {request.model}")

        model_config = config.get("models", {}).get(request.model)

        if not model_config:
            available_models = list(config.get("models", {}).keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found in configuration. Available models: {available_models}"
            )

        model_path = model_config.get("model_path")
        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' has no model_path configured"
            )

        # Get vlm_type for multimodal support (None for LLM-only models)
        vlm_type = model_config.get("vlm_type")

        # Apply chat template
        formatted_prompt = await apply_chat_template_for_model(
            model_path=model_path,
            messages=request.messages,
            tools=request.tools,
            add_generation_prompt=request.add_generation_prompt,
            chat_template_kwargs=request.chat_template_kwargs,
            vlm_type=vlm_type
        )

        logger.info(
            f"[apply-template] model={request.model} messages={len(request.messages)} "
            f"tools={len(request.tools) if request.tools else 0} "
            f"add_generation_prompt={request.add_generation_prompt} "
            f"vlm_type={vlm_type or 'none'} "
            f"prompt_length={len(formatted_prompt)}"
        )

        return {"prompt": formatted_prompt}

    except ValueError as exc:
        # User error (bad model, bad path, etc.)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # Template application error
        raise HTTPException(status_code=500, detail=str(exc))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        # Unexpected error
        logger.error(f"[apply-template] Unexpected error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Template application failed: {str(exc)}")


@app.post("/v1/apply-template", dependencies=[Depends(verify_api_key)])
async def apply_template_v1(request: ApplyTemplateRequest):
    """OpenAI-style path alias for /apply-template."""
    return await apply_template(request)


@app.post("/tokenize", dependencies=[Depends(verify_api_key)])
async def tokenize(request: TokenizeRequest):
    """
    Tokenize text content and return token IDs.

    This endpoint tokenizes text using the model's tokenizer and returns
    a list of token IDs. Useful for token counting, prompt analysis, and benchmarking.

    Returns:
        {"tokens": [list of token IDs]}

    Raises:
        400: Invalid request (missing content, bad model, etc.)
        500: Tokenization failed
    """
    try:
        # Load config to get model_path
        config = _load_openarc_config()

        # If model not specified, use first available model
        if not request.model:
            models = list(config.get("models", {}).keys())
            if not models:
                raise HTTPException(status_code=400, detail="No models configured")
            request.model = models[0]
            logger.info(f"[tokenize] No model specified, using first available: {request.model}")

        model_config = config.get("models", {}).get(request.model)
        if not model_config:
            available_models = list(config.get("models", {}).keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found in configuration. Available models: {available_models}"
            )

        model_path = model_config.get("model_path")
        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' has no model_path configured"
            )

        # Tokenize content
        tokens = await tokenize_content(
            model_path=model_path,
            content=request.content,
            add_special=request.add_special
        )

        logger.info(
            f"[tokenize] model={request.model} content_len={len(request.content)} "
            f"tokens={len(tokens)} add_special={request.add_special}"
        )

        return {"tokens": tokens}

    except ValueError as exc:
        # User error (bad model, bad path, etc.)
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        # Unexpected error
        logger.error(f"[tokenize] Unexpected error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(exc)}")


@app.post("/v1/tokenize", dependencies=[Depends(verify_api_key)])
async def tokenize_v1(request: TokenizeRequest):
    """OpenAI-style path alias for /tokenize."""
    return await tokenize(request)