# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better. 

import datetime
import json
import logging
import os
import time
import uuid
import traceback
from typing import Any, AsyncIterator, List, Optional, Dict, Union

from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.server.model_registry import ModelLoadConfig, ModelRegistry, ModelUnloadConfig
from src.server.worker_registry import WorkerRegistry
from src.server.models.openvino import OV_KokoroGenConfig
from src.server.models.ov_genai import OVGenAI_GenConfig, OVGenAI_WhisperGenConfig
from src.server.models.optimum import PreTrainedTokenizerConfig, RerankerConfig

#===============================================================#
# Logging
#===============================================================#


logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



#===============================================================#
# FastAPI configuration
#===============================================================#

app = FastAPI()

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
# OpenArc internal
#===============================================================#

_registry = ModelRegistry()
_workers = WorkerRegistry(_registry)


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
    task:Optional[str] = None
    config: Optional[PreTrainedTokenizerConfig] = None #not implemented

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
        logger.info(f"[chat/completions] Received tools: {request.tools}")
        
        config_kwargs = {
            "messages": request.messages,
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
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
        
        logger.info(f"[chat/completions] config_kwargs tools: {config_kwargs.get('tools', 'NOT PRESENT')}")

        generation_config = OVGenAI_GenConfig(**config_kwargs)
        logger.info(f"[chat/completions] generation_config.tools: {generation_config.tools}")

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

                # Final stop signal per OpenAI SSE with usage
                prompt_tokens = (metrics_data or {}).get("input_token", 0)
                completion_tokens = (metrics_data or {}).get("new_token", 0)
                total_tokens = (metrics_data or {}).get("total_token", prompt_tokens + completion_tokens)
                final_payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
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

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
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
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}")




@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def openai_audio_transcriptions(request: OpenAIWhisperRequest):
    try:
        gen_config = OVGenAI_WhisperGenConfig(audio_base64=request.audio_base64)
        result = await _workers.transcribe_whisper(request.model, gen_config)
        return {"text": result.get("text", ""), "metrics": result.get("metrics", {})}
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
        if request.config:

            tok_config = PreTrainedTokenizerConfig.model_validate(request.config)
            base_data = tok_config.model_dump()
            rr_config = RerankerConfig.model_validate(base_data | {"query":request.query,"documents":request.documents})
        if request.prefix:
            rr_config.prefix = request.prefix
        if request.suffix:
            rr_config.suffix = request.suffix
        if request.task:
            rr_config.task = request.task
            
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