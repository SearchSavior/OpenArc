import os

import uvicorn
import logging
from pathlib import Path

# Configure logging. Default to openarc.log in project root, but can be overridden with OPENARC_LOG_FILE env var.
# Setting this to /dev/null effectively disables file logging if desired.
default_log_file = Path(__file__).parent.parent.parent.parent / "openarc.log"
log_file = Path(os.getenv("OPENARC_LOG_FILE", default_log_file))

def _level_from_verbose(verbose: int) -> str:
    # our own code (src.* and OpenArc loggers)
    if verbose >= 3:
        return "DEBUG"
    if verbose >= 2:
        return "INFO"
    if verbose == 1:
        return "WARNING"
    return "ERROR"


def _access_level_from_verbose(verbose: int) -> str:
    # http request logs (openarc.access + uvicorn.access). shown from -vv.
    if verbose >= 4:
        return "DEBUG"
    if verbose >= 2:
        return "INFO"
    return "WARNING"


def _root_level_from_verbose(verbose: int) -> str:
    # floor for third-party libraries (httpx, transformers, openvino, ...).
    # kept above our own level so -vvv shows our debug without library noise;
    # -vvvv drops it to DEBUG too.
    if verbose >= 4:
        return "DEBUG"
    if verbose >= 2:
        return "INFO"
    if verbose == 1:
        return "WARNING"
    return "ERROR"


def _build_log_config(verbose: int):
    app_level = _level_from_verbose(verbose)
    access_level = _access_level_from_verbose(verbose)
    root_level = _root_level_from_verbose(verbose)
    uvicorn_level = "DEBUG" if verbose >= 4 else "INFO"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
            "access": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": str(log_file),
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access_file": {
                "formatter": "access",
                "class": "logging.FileHandler",
                "filename": str(log_file),
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default", "file"],
                "level": uvicorn_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "level": uvicorn_level,
                "handlers": ["default", "file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access", "access_file"],
                "level": access_level,
                "propagate": False,
            },
            # our own application code. set explicitly so -vvv shows our debug
            # logs while the root floor keeps third-party libraries quiet.
            "src": {
                "level": app_level,
                "propagate": True,
            },
            "OpenArc": {
                "level": app_level,
                "propagate": True,
            },
            "openarc.access": {
                "handlers": ["default", "file"],
                "level": access_level,
                "propagate": False,
            },
        },
        "root": {
            "level": root_level,
            "handlers": ["default", "file"],
        },
    }


logger = logging.getLogger("OpenArc")

def start_server(host: str = "0.0.0.0", port: int = 8001, reload: bool = False, verbose: int = 0):
    """
    Launches the OpenArc API server

    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """

    # applies only until uvicorn.run() installs the dict config below.
    logger.setLevel(getattr(logging, _level_from_verbose(verbose)))
    logging.getLogger().setLevel(getattr(logging, _root_level_from_verbose(verbose)))

    print(f"Launching  {host}:{port}")
    print("--------------------------------")
    print("OpenArc endpoints:")
    print("  - POST   /openarc/load           Load a model")
    print("  - POST   /openarc/unload         Unload a model")
    print("  - GET    /openarc/status         Get model status")
    print("  - GET    /openarc/metrics            Get hardware telemetry")
    print("  - POST   /openarc/models/update      Update model configuration")
    print("  - POST   /openarc/bench              Run inference benchmark")
    print("  - GET    /openarc/downloader         List active model downloads")
    print("  - POST   /openarc/downloader         Start a model download")
    print("  - DELETE /openarc/downloader         Cancel a model download")
    print("  - POST   /openarc/downloader/pause   Pause a model download")
    print("  - POST   /openarc/downloader/resume  Resume a model download")
    print("--------------------------------")
    print("OpenAI compatible endpoints:")
    print("  - GET    /v1/models")
    print("  - POST   /v1/chat/completions")
    print("  - POST   /v1/audio/transcriptions: Whisper only")
    print("  - POST   /v1/audio/speech: Kokoro only")
    print("  - POST   /v1/embeddings")
    print("  - POST   /v1/rerank")


    uvicorn.run(
        "src.server.main:app",
        host=host,
        port=port,
        log_config=_build_log_config(verbose),
        reload=reload
    )
