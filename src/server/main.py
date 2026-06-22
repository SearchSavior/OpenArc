# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better.

import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.cli.utils import get_config_file_path
from starlette.middleware.base import BaseHTTPMiddleware

from src.server.deps import _registry
from src.server.models.registration import ModelLoadConfig
from src.server.routes.openai import router as openai_router
from src.server.routes.openarc import router as openarc_router

logger = logging.getLogger(__name__)
_access_logger = logging.getLogger("openarc.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        _access_logger.info(
            f"Request received: {request.method} {request.url.path} from {client_ip}"
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            _access_logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} duration={process_time:.3f}s"
            )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            _access_logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"error={str(e)} duration={process_time:.3f}s"
            )
            raise


async def monitor_power_state(registry):
    import psutil
    import asyncio
    from src.server.models.registration import ModelLoadConfig
    
    logger.info("Power monitor background task started.")
    last_plugged = None
    
    while True:
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                plugged = battery.power_plugged
                if last_plugged is not None and plugged != last_plugged:
                    target_device = "GPU" if plugged else "NPU"
                    logger.info(f"Power status changed (plugged={plugged}). Switching LLM models to {target_device}...")
                    
                    async with registry._lock:
                        loaded_models = list(registry._models.values())
                    
                    for record in loaded_models:
                        if record.model_type == "llm" and record.device in ("GPU", "NPU", "AUTO", "GPU.0", "GPU.1"):
                            logger.info(f"Power Switch: Unloading '{record.model_name}' from {record.device}...")
                            load_config = ModelLoadConfig(
                                model_path=record.model_path,
                                model_name=record.model_name,
                                model_type=record.model_type,
                                engine=record.engine,
                                device=target_device,
                                runtime_config=record.runtime_config,
                                cache_dir=getattr(record, "cache_dir", None),
                                draft_model_path=getattr(record, "draft_model_path", None),
                                draft_device=getattr(record, "draft_device", "CPU"),
                                num_assistant_tokens=getattr(record, "num_assistant_tokens", None),
                                assistant_confidence_threshold=getattr(record, "assistant_confidence_threshold", None),
                            )
                            try:
                                await registry.register_unload(record.model_name)
                                await registry.register_load(load_config)
                                logger.info(f"Power Switch: Loaded '{record.model_name}' on {target_device} successfully!")
                            except Exception as err:
                                logger.error(f"Power Switch: Failed to switch '{record.model_name}' to {target_device}: {err}")
                last_plugged = plugged
        except Exception as e:
            logger.error(f"Error in power monitor: {e}")
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    power_task = None
    if os.getenv("OPENARC_POWER_MONITOR", "").strip().lower() == "true":
        import asyncio
        power_task = asyncio.create_task(monitor_power_state(_registry))

    models = os.getenv("OPENARC_STARTUP_MODELS", "").strip()
    if models:
        from pathlib import Path

        config_file = get_config_file_path()
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

            for name in models.split(","):
                name = name.strip()
                model_config = config.get("models", {}).get(name)
                if not model_config:
                    logger.warning(f"Startup: model '{name}' not in config, skipping")
                    continue
                
                model_path = model_config.get("model_path")
                if model_path and not Path(model_path).is_absolute():
                    model_config["model_path"] = str((config_file.parent / model_path).resolve())

                cache_dir = model_config.get("cache_dir")
                if cache_dir and not Path(cache_dir).is_absolute():
                    cache_dir = str((config_file.parent / cache_dir).resolve())
                    model_config["cache_dir"] = cache_dir

                try:
                    if cache_dir:
                        # Create the cache directory at startup if it doesn't exist.
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    await _registry.register_load(ModelLoadConfig(**model_config))
                    logger.info(f"Startup: loaded '{name}'")
                except Exception as e:
                    logger.error(f"Startup: failed to load '{name}': {e}")

    try:
        yield
    finally:
        if power_task:
            power_task.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}", exc_info=True)
    return JSONResponse(
        status_code=422, content={"status": "error", "detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    logger.error(f"Full traceback:\n{''.join(traceback.format_tb(exc.__traceback__))}")
    return JSONResponse(
        status_code=500, content={"status": "error", "detail": str(exc)}
    )


app.include_router(openarc_router)
app.include_router(openai_router)
