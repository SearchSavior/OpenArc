import uvicorn
import logging
# from src.api.optimum_api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("OpenArc-Server")

def start_server(host: str = "0.0.0.0", openarc_port: int = 8001, reload: bool = False):
    """
    Launches the OpenArc API server
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    logger.info(f"Launching OpenArc on {host}:{openarc_port}")
    logger.info("Available endpoints:")
    logger.info("  - POST   optimum/model/load      Load a model")
    logger.info("  - DELETE optimum/model/unload    Unload current model")
    logger.info("  - GET    optimum/status         Get model status")
    logger.info("  - GET    optimum/docs           API documentation")
    logger.info("  - POST   /v1/chat/completions openai compatible endpoint")
    logger.info("  - GET    /v1/models     openai compatible endpoint")
    
    
    # Start the server
    uvicorn.run(
        "src.api.optimum_api:app",
        host=host,
        port=openarc_port,
        log_level="info"
    )
