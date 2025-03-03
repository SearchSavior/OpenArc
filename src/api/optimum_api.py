from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import StreamingResponse, JSONResponse

from typing import Optional, AsyncIterator, List, Dict
from pydantic import BaseModel, Extra
from datetime import datetime

import logging
import time
import uuid
import json

from src.engine.optimum.optimum_inference_core import (
    OV_LoadModelConfig,
    OV_Config,
    OV_GenerationConfig,
    Optimum_InferenceCore,
)

app = FastAPI(title="OpenVINO Inference API")

# Global state to store model instance
model_instance: Optional[Optimum_InferenceCore] = None

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "default"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

    class Config:
        extra = Extra.ignore


@app.post("/optimum/model/load")
async def load_model(load_config: OV_LoadModelConfig, ov_config: OV_Config):
    """Load a model with the specified configuration"""
    global model_instance
    try:
        # Initialize new model
        model_instance = Optimum_InferenceCore(
            load_model_config=load_config,
            ov_config=ov_config,
        )
        
        # Load the model
        model_instance.load_model()
        
        return {"status": "success", "message": f"Model {load_config.id_model} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/optimum/model/unload")
async def unload_model():
    """Unload the current model"""
    global model_instance
    if model_instance:
        model_instance.util_unload_model()
        model_instance = None
        return {"status": "success", "message": "Model unloaded successfully"}
    return {"status": "success", "message": "No model was loaded"}

@app.post("/optimum/generate")
async def generate_text(generation_config: OV_GenerationConfig):
    """Generate text either as a stream or a full response, based on the stream field"""
    global model_instance
    if not model_instance:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    # Check if the client requested streaming
    if generation_config.stream:
        async def text_stream() -> AsyncIterator[str]:
            async for token in model_instance.generate_stream(generation_config):
                yield token
        return StreamingResponse(text_stream(), media_type="text/event-stream")
    else:
        try:
            generated_text, metrics = model_instance.generate_text(generation_config)
            return {
                "generated_text": generated_text,
                "performance_metrics": metrics
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimum/status")
async def get_status():
    """Get current model status and performance metrics"""
    global model_instance
    if not model_instance:
        return {
            "status": "no_model",
            "id_model": None,
            "device": None
        }
    
    return {
        "status": "loaded",
        "id_model": model_instance.load_model_config.id_model,
        "device": model_instance.load_model_config.device
    }

########################################################
# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better. 


# OpenAI-like API Endpoints for using with different frontends
# These require other features to be implemented like performance tracking,etc


@app.get("/v1/models")
async def get_models():
    """Get list of available models in openai format"""
    global model_instance
    data = []

    if model_instance:
        model_data = {
            "id": model_instance.load_model_config.id_model,
            "object": "model",
            "created": int(datetime.utcnow().timestamp()),
            "owned_by": "OpenArc",  # Our platform identifier a placeholder
        }
        data.append(model_data)

    return {
        "object": "list",
        "data": data
    }

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatCompletionRequest):
    global model_instance
    if not model_instance:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Toggle debug output here
    DEBUG = True
    if DEBUG:
        print("\n=== Received Request ===")
        print("Raw messages:", request.messages)
        print("Params - temperature:", request.temperature)
        print("Params - max_tokens:", request.max_tokens)

    try:
        # Convert OpenAI-style messages to conversation format
        conversation = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in request.messages
        ]

        if DEBUG:
            print("Processed conversation:", conversation)

        # Create generation config with conversation structure
        generation_config = OV_GenerationConfig(
            conversation=conversation,  # This matches the original working format
            temperature=request.temperature or 0.7,
            max_new_tokens=request.max_tokens or 512, # Handles both max_tokens and max_new_tokens via alias
            stop_sequences=request.stop or [],
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=1
        )

        if request.stream:
            async def stream_generator():
                try:
                    async for token in model_instance.generate_stream(generation_config):
                        # Properly escape the content for JSON, preserving all whitespace
                        escaped_token = json.dumps(token)[1:-1]  # Remove surrounding quotes
                        yield f"data: {{\"object\": \"chat.completion.chunk\", \"choices\": [{{\"delta\": {{\"content\": \"{escaped_token}\"}}}}]}}\n\n"

                except Exception as e:
                    print(f"Error during streaming: {str(e)}")
                finally:
                    yield "data: [DONE]\n\n"


            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            generated_text, metrics = model_instance.generate_text(generation_config)
            print(metrics)
            return JSONResponse(content={
                "id": f"ov-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_instance.load_model_config.id_model,
                "choices": [{
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "length"
                }],
                "timings": {
                    "prompt_tokens": metrics.get("input_tokens", 0),
                    "completion_tokens": metrics.get("output_tokens", 0),
                    "total_tokens": metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0)
                }
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))