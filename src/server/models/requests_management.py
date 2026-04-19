from typing import Optional
from pydantic import BaseModel


class DownloaderRequest(BaseModel):
    model_name: str
    path: Optional[str] = None


class DownloaderActionRequest(BaseModel):
    model_name: str
