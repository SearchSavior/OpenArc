from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from typing import AsyncIterator
from src2.api.model_registry import ModelRegistry, ModelLoadConfig, ModelUnloadConfig
from src2.api.worker_registry import WorkerRegistry
from src2.api.base_config import OVGenAI_TextGenConfig



app = FastAPI()

_registry = ModelRegistry()
_workers = WorkerRegistry(_registry)


@app.post("/openarc/load")
async def load_model(load_config: ModelLoadConfig):
    try:
        model_id = await _registry.register_load(load_config)
        return {"model_id": model_id, "model_name": load_config.model_name, "status": "loaded"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(exc)}")

@app.post("/openarc/unload")
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

@app.get("/openarc/status")
async def get_status():
    """Get registry status showing all loaded models."""
    return await _registry.status()


class GenerateRequest(BaseModel):
    model_name: str
    gen_config: OVGenAI_TextGenConfig


@app.post("/openarc/generate")
async def generate_text(req: GenerateRequest):
    """Generate text using a loaded model. Supports streaming and non-streaming."""
    try:
        if req.gen_config.stream:
            async def event_stream() -> AsyncIterator[bytes]:
                async for item in _workers.stream_generate(req.model_name, req.gen_config):
                    if isinstance(item, dict):
                        payload = {"event": "metrics", "data": item}
                        yield (f"data: {json.dumps(payload)}\n\n").encode()
                    else:
                        yield (f"data: {item}\n\n").encode()

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            result = await _workers.generate(req.model_name, req.gen_config)
            return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
