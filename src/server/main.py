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
from starlette.middleware.base import BaseHTTPMiddleware

from src.server.deps import _registry
from src.server.models.registration import ModelLoadConfig
from src.server.routes.openai import router as openai_router
from src.server.routes.openarc import router as openarc_router

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            f"Request received: {request.method} {request.url.path} from {client_ip}"
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} duration={process_time:.3f}s"
            )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"error={str(e)} duration={process_time:.3f}s"
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
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
