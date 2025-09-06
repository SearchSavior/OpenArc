from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src2.api.model_registry import ModelRegistry, ModelLoadConfig, ModelUnloadConfig



app = FastAPI()

_registry = ModelRegistry()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
