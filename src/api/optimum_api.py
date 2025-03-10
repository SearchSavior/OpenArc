# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better. 


from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional, AsyncIterator, List, Dict
from pydantic import BaseModel
from datetime import datetime

import warnings
import logging
import time
import uuid
import json
import os


from src.engine.optimum.optimum_inference_core import (
    OV_LoadModelConfig,
    OV_Config,
    OV_GenerationConfig,
    Optimum_InferenceCore,
)

# Suppress specific deprecation warnings from optimum/numpy
# This blocks warning the clog the API logs 
warnings.filterwarnings("ignore", message="__array__ implementation doesn't accept a copy keyword")

app = FastAPI(title="OpenVINO Inference API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Global state to store model instance
model_instance: Optional[Optimum_InferenceCore] = None

logger = logging.getLogger("optimum_api")
logger.setLevel(logging.DEBUG)

# API key authentication
API_KEY = os.getenv("OPENARC_API_KEY")
security = HTTPBearer()

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "default"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

    class Config:
        extra = "ignore"

class CompletionRequest(BaseModel):
    prompt: str
    model: str = "default"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

    class Config:
        extra = "ignore"

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

@app.post("/optimum/model/load", dependencies=[Depends(verify_api_key)])
async def load_model(load_config: OV_LoadModelConfig, ov_config: OV_Config):
    """Load a model with the specified configuration"""
    global model_instance
    logger.info("POST /optimum/model/load called with load_config: %s, ov_config: %s", load_config, ov_config)
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

@app.delete("/optimum/model/unload", dependencies=[Depends(verify_api_key)])
async def unload_model():
    """Unload the current model"""
    global model_instance
    logger.info("DELETE /optimum/model/unload called")
    if model_instance:
        model_instance.util_unload_model()
        model_instance = None
        return {"status": "success", "message": "Model unloaded successfully"}
    return {"status": "success", "message": "No model was loaded"}

@app.post("/optimum/generate", dependencies=[Depends(verify_api_key)])
async def generate_text(generation_config: OV_GenerationConfig):
    """Generate text either as a stream or a full response, based on the stream field"""
    global model_instance
    logger.info("POST /optimum/generate called with generation_config: %s", generation_config)
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

@app.get("/optimum/status", dependencies=[Depends(verify_api_key)])
async def get_status():
    """Get current model status and performance metrics"""
    global model_instance
    logger.info("GET /optimum/status called")
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


# OpenAI-like API


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def get_models():
    """Get list of available models in openai format"""
    global model_instance
    logger.info("GET /v1/models called")
    data = []

    if model_instance:
        model_data = {
            "id": model_instance.load_model_config.id_model,
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
    global model_instance
    if not model_instance:
        logger.error("POST /v1/chat/completions failed: No model loaded")
        raise HTTPException(status_code=503, detail="No model loaded")
    logger.info("POST /v1/chat/completions called with messages: %s", request.messages)

    # Toggle debug output here
    DEBUG = True
    if DEBUG:
        print("\n=== Received Request ===")
        print("Raw messages:", request.messages)
        print("Params - temperature:", request.temperature)
        print("Params - max_tokens:", request.max_tokens)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(request.model)
            prompt_len=len(tokenizer.apply_chat_template(request.messages))
        except Exception as e:
            print(f"Token counting error: {e}")

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
                # Performance tracking variables
                start_time = time.perf_counter()
                first_token_time = None
                token_count = 0
                try:
                    async for token in model_instance.generate_stream(generation_config):
                        token_count += 1

                        # Record time of first token
                        if token_count == 1:
                            first_token_time = time.perf_counter()
                            eval_time = first_token_time - start_time

                        # Properly escape the content for JSON, preserving all whitespace
                        escaped_token = json.dumps(token)[1:-1]  # Remove surrounding quotes
                        yield f"data: {{\"object\": \"chat.completion.chunk\", \"choices\": [{{\"delta\": {{\"content\": \"{escaped_token}\"}}}}]}}\n\n"

                except Exception as e:
                    print(f"Error during streaming: {str(e)}")
                finally:
                    # Calculate final metrics
                    end_time = time.perf_counter()
                    total_time = end_time - start_time

                    if first_token_time:
                        tokens_per_second = token_count / (end_time - start_time)
                        eval_tokens_per_second = prompt_len / eval_time

                        if DEBUG:
                            print("\n=== Streaming Performance ===")
                            print(f"Total generation time: {total_time:.3f} seconds")
                            print(f"Prompt evaluation: {prompt_len} tokens in {eval_time:.3f} seconds ({eval_tokens_per_second:.2f} T/s)")
                            print(f"Response generation: {token_count} tokens in ({tokens_per_second:.2f} T/s)")

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

@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
async def openai_completions(request: CompletionRequest):
    global model_instance
    if not model_instance:
        logger.error("POST /v1/completions failed: No model loaded")
        raise HTTPException(status_code=503, detail="No model loaded")
    logger.info("POST /v1/completions called with prompt: %s", request.prompt)

    # Convert prompt into conversation format (single user message)
    conversation = [{"role": "user", "content": request.prompt}]

    # Create generation config
    generation_config = OV_GenerationConfig(
        conversation=conversation,
        temperature=request.temperature or 0.7,
        max_new_tokens=request.max_tokens or 8192,
        stop_sequences=request.stop or [],
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1
    )

    # Handle streaming response
    if request.stream:
        async def stream_generator():
            async for token in model_instance.generate_stream(generation_config):
                # Properly escape and format for SSE
                escaped_token = json.dumps(token)[1:-1]
                yield f"data: {{\"object\": \"text_completion.chunk\", \"choices\": [{{\"text\": \"{escaped_token}\"}}]}}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Handle regular response
    try:
        generated_text, metrics = model_instance.generate_text(generation_config)
        return JSONResponse(content={
            "id": f"ov-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_instance.load_model_config.id_model,
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