import uvicorn
import logging
from pathlib import Path

# Configure logging
log_file = Path(__file__).parent.parent.parent.parent / "openarc.log"

def _level_from_verbose(verbose: int) -> str:
    if verbose >= 2:
        return "INFO"
    if verbose == 1:
        return "WARNING"
    return "ERROR"


def _build_log_config(verbose: int):
    app_level = _level_from_verbose(verbose)
    access_level = "INFO" if verbose >= 3 else "WARNING"

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
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["default", "file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access", "access_file"],
                "level": access_level,
                "propagate": False,
            },
            "openarc.access": {
                "handlers": ["default", "file"],
                "level": access_level,
                "propagate": False,
            },
        },
        "root": {
            "level": app_level,
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

    app_level_name = _level_from_verbose(verbose)
    app_level_num = getattr(logging, app_level_name)

    logger.setLevel(app_level_num)
    logging.getLogger().setLevel(app_level_num)

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
