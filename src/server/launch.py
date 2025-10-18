import uvicorn
import logging
from pathlib import Path

# Configure logging
log_file = Path(__file__).parent.parent.parent / "openarc.log"

# Create a custom logging configuration for uvicorn
LOG_CONFIG = {
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
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file"],
    },
}

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger("OpenArc")

def start_server(host: str = "0.0.0.0", openarc_port: int = 8001, reload: bool = False):
    """
    Launches the OpenArc API server
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    logger.info(f"Launching  {host}:{openarc_port}")
    logger.info("--------------------------------")
    logger.info("OpenArc endpoints:")
    logger.info("  - POST   /openarc/load           Load a model")
    logger.info("  - POST   /openarc/unload         Unload a model")
    logger.info("  - GET    /openarc/status         Get model status")
    logger.info("--------------------------------")
    logger.info("OpenAI compatible endpoints:")
    logger.info("  - GET    /v1/models")
    logger.info("  - POST   /v1/chat/completions")
    logger.info("  - POST   /v1/audio/transcriptions: Whisper only")
    logger.info("  - POST   /v1/audio/speech: Kokoro only")
    logger.info("  - POST   /v1/embeddings")
    

    uvicorn.run(
        "src.server.main:app",
        host=host,
        port=openarc_port,
        log_config=LOG_CONFIG,
        reload=reload
    )