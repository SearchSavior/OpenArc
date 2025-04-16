# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better. 

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional, List, Any
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path

import warnings
import logging
import time
import uuid
import json
import os

from src.engine.optimum.optimum_base_config import (
    OV_LoadModelConfig,
    OV_Config,
    OV_GenerationConfig,
    create_optimum_model,
    ModelType
)


# Suppress specific deprecation warnings from optimum implementation of numpy arrays
# This block prevents clogging the API logs 
warnings.filterwarnings("ignore", message="__array__ implementation doesn't accept a copy keyword")

app = FastAPI(title="OpenArc API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Global state to store multiple model instances
model_instances = {}

logger = logging.getLogger("optimum_api")

# API key authentication
API_KEY = os.getenv("OPENARC_API_KEY")
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key provided in the Authorization header"""
    if credentials.credentials != API_KEY:
        logger.warning(f"Invalid API key: {credentials.credentials}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def get_final_model_id(model_id: str) -> str:
    """Extracts the final segment of the model id path so we dont display the whole path."""
    return Path(model_id).name

@app.post("/optimum/model/load", dependencies=[Depends(verify_api_key)])
async def load_model(load_config: OV_LoadModelConfig, ov_config: OV_Config):
    """Load a model with the specified configuration"""
    global model_instances
    logger.info("POST /optimum/model/load called with load_config: %s, ov_config: %s", load_config, ov_config)
    try:
        # Initialize new model using the factory function
        new_model = create_optimum_model(
            load_model_config=load_config,
            ov_config=ov_config
        )
        
        # Load the model
        new_model.load_model()
        
        # Store the model instance with its ID as the key
        model_id = get_final_model_id(load_config.id_model)
        model_instances[model_id] = new_model
        
        return {"status": "success", "message": f"Model {model_id} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/optimum/model/unload", dependencies=[Depends(verify_api_key)])
async def unload_model(model_id: str):
    """Unload the current model"""
    global model_instances
    logger.info(f"DELETE /optimum/model/unload called for model {model_id}")
    if model_id in model_instances:
        model_instances[model_id].util_unload_model()
        del model_instances[model_id]
        return {"status": "success", "message": "Model unloaded successfully"}
    return {"status": "success", "message": f"Model {model_id} was not loaded"}

@app.get("/optimum/status", dependencies=[Depends(verify_api_key)])
async def get_status():
    """Get current model status and performance metrics"""
    global model_instances
    logger.info("GET /optimum/status called")
    loaded_models = {}
    for model_id, model in model_instances.items():
        loaded_models[model_id] = {
            "status": "loaded",
            "device": model.load_model_config.device,
            "model_metadata": model.model_metadata
        }
    
    return {
        "loaded_models": loaded_models,
        "total_models_loaded": len(model_instances)
    }


# OpenAI-like API

class ChatCompletionRequest(BaseModel):
    messages: Any
    model: str = "default"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    num_return_sequences: Optional[int] = None


class CompletionRequest(BaseModel):
    prompt: str
    model: str = "default"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    num_return_sequences: Optional[int] = None  


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def get_models():
    """Get list of available models in openai format"""
    global model_instances
    logger.info("GET /v1/models called")
    data = []

    for model_id, model in model_instances.items():
        model_data = {
            "id": model_id,
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "OpenArc", 
        }
        data.append(model_data)

    return {
        "object": "list",
        "data": data
    }

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def openai_chat_completions(request: ChatCompletionRequest):
    global model_instances
    model_id = get_final_model_id(request.model)
    
    if model_id not in model_instances:
        logger.error("POST /v1/chat/completions failed: No model loaded")
        raise HTTPException(status_code=503, detail=f"Model {model_id} not loaded")
        
    model_instance = model_instances[model_id]
    logger.info("POST /v1/chat/completions called with messages: %s", request.messages)

    try:
        # Handle vision model messages differently
        if model_instance.model_metadata["model_type"] == ModelType.VISION:
            conversation = []
            for msg in request.messages:
                if isinstance(msg["content"], list):
                    # Handle multimodal content (text + images)
                    vision_message = {
                        "role": msg["role"],
                        "content": msg["content"]  # Keep the full content structure for vision models
                    }
                    conversation.append(vision_message)
                else:
                    # Handle text-only messages
                    conversation.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        else:
            # Regular text model handling
            conversation = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in request.messages
            ]

        # Build config dict, only include non-None values
        config_kwargs = {
            "conversation": conversation,
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.do_sample,
            "num_return_sequences": request.num_return_sequences,
            "stream": request.stream,
            # Note: stop_sequences is not part of OV_GenerationConfig, handled separately if needed
        }
        # Remove keys with value None
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        # Create generation config with filtered arguments
        generation_config = OV_GenerationConfig(**config_kwargs)

        if request.stream:
            async def stream_generator():
                current_metrics = None
                try:
                    if model_instance.model_metadata["model_type"] == ModelType.VISION:
                        stream_method = model_instance.generate_vision_stream
                    elif model_instance.model_metadata["model_type"] == ModelType.TEXT:
                        stream_method = model_instance.generate_stream

                    async for token_chunk, metrics_chunk in stream_method(generation_config):
                        if token_chunk is not None:
                            # Stream the token chunk
                            escaped_token = json.dumps(token_chunk)[1:-1]
                            yield f"data: {{\"object\": \"chat.completion.chunk\", \"choices\": [{{\"delta\": {{\"content\": \"{escaped_token}\"}}}}]}}\n\n"
                        if metrics_chunk is not None:
                            # Store the final metrics when received
                            current_metrics = metrics_chunk

                except Exception as e:
                    logger.error(f"Error during streaming: {str(e)}", exc_info=True) # Log traceback
                finally:
                    if current_metrics:
                        # Log the full metrics dictionary as structured JSON
                        logger.info(f"Performance metrics: {json.dumps(current_metrics, indent=2)}")
                    yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            # For non-streaming responses, use the appropriate generate method based on model type
            model_type = model_instance.model_metadata["model_type"]
            if model_type == ModelType.VISION:
                # Call the new vision-specific non-streaming method
                generated_text, metrics = model_instance.generate_vision_text(generation_config)
            elif model_type == ModelType.TEXT:
                generated_text, metrics = model_instance.generate_text(generation_config)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model type '{model_type}' for chat completions.")

            # Log metrics server-side for non-streaming requests
            if metrics:
                 logger.info(f"Performance metrics (non-streaming): {json.dumps(metrics, indent=2)}")

            return JSONResponse(content={
                "id": f"ov-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "length"
                }],
                "performance": metrics,
                "timings": {
                    "prompt_tokens": metrics.get("input_tokens", 0),
                    "completion_tokens": metrics.get("output_tokens", 0),
                    "total_tokens": metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0)
                }
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
async def openai_completions(request: CompletionRequest):
    global model_instances
    model_id = get_final_model_id(request.model)
    
    if model_id not in model_instances:
        logger.error("POST /v1/completions failed: No model loaded")
        raise HTTPException(status_code=503, detail=f"Model {model_id} not loaded")
        
    model_instance = model_instances[model_id]
    logger.info("POST /v1/completions called with prompt: %s", request.prompt)

    # Convert prompt into conversation format (single user message)
    conversation = [{"role": "user", "content": request.prompt}]

    # Create generation config
    generation_config = OV_GenerationConfig(
        conversation=conversation,
        temperature=request.temperature or 0.7,
        max_new_tokens=request.max_tokens or 8192,
        stop_sequences=request.stop or [],
        top_p=request.top_p or 0.9,  # default value for top_p
        top_k=request.top_k or 50,   # default value for top_k
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1
    )

    # Use model type to determine which generation method to use
    model_type = model_instance.model_metadata["model_type"]

    # Handle streaming response
    if request.stream:
        async def stream_generator():
            # Route to the appropriate stream generator based on model type
            if model_type == ModelType.VISION:
                stream_method = model_instance.generate_vision_stream
            else:
                stream_method = model_instance.generate_stream
                
            async for token in stream_method(generation_config):
                # Properly escape and format for SSE
                escaped_token = json.dumps(token)[1:-1]
                yield f"data: {{\"object\": \"text_completion.chunk\", \"choices\": [{{\"text\": \"{escaped_token}\"}}]}}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Handle regular response
    try:
        # For non-streaming responses, use the appropriate generate method
        if model_type == ModelType.VISION:
            generated_text, metrics = model_instance.generate_text(generation_config)
        elif model_type == ModelType.TEXT:
            generated_text, metrics = model_instance.generate_text(generation_config)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model type '{model_type}' for completions endpoint. Only VISION and TEXT types are supported."
            )
            
        return JSONResponse(content={
            "id": f"ov-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": metrics.get("input_tokens", 0),
                "completion_tokens": metrics.get("output_tokens", 0),
                "total_tokens": metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0)
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
