from fastapi import FastAPI, HTTPException

from src2.api.model_registry import ModelRegistry, ModelLoadConfig


app = FastAPI()

_registry = ModelRegistry()


@app.post("/openarc/load")
async def load_model(load_config: ModelLoadConfig):
    try:
        model_id = await _registry.register_load(load_config)
        return {
            "model_id": model_id, 
            "model_name": load_config.model_name, 
            "status": "loading",
            "message": "Model loading started in background. Check /openarc/status for progress."
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start model loading: {str(exc)}")

@app.post("/openarc/unload")
async def unload_model():
    raise HTTPException(status_code=501, detail="Not implemented")

@app.get("/openarc/status")
async def get_status():
    """Get registry status showing all loaded models."""
    return await _registry.status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
