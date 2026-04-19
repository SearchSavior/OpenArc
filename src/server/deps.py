import logging
import os

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.server.model_registry import ModelRegistry
from src.server.worker_registry import WorkerRegistry

logger = logging.getLogger(__name__)

_registry = ModelRegistry()
_workers = WorkerRegistry(_registry)

API_KEY = os.getenv("OPENARC_API_KEY")
AUTH_REQUIRED = os.getenv("OPENARC_API_KEY_REQUIRED", "false").lower() == "true"
security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not AUTH_REQUIRED:
        return None
    if credentials is None or credentials.credentials != API_KEY:
        logger.error(
            f"Invalid API key: {credentials.credentials if credentials else 'missing'}"
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
