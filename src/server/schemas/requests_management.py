from pydantic import BaseModel


class DownloaderRequest(BaseModel):
    model_name: str


class DownloaderActionRequest(BaseModel):
    model_name: str
