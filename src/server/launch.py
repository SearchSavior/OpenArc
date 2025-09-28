import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
    

    uvicorn.run(
        "src.server.main:app",
        host=host,
        port=openarc_port,
        log_level="info",
        reload=reload
    )