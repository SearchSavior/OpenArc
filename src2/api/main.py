from fastapi import FastAPI



from src2.api.base_config import (
    OVGenAI_TextGenConfig, 
    OVGenAI_LoadConfig
)
from src2.engine.text2text import OVGenAI_Text2Text



@app.post("/model/load")
def load_model(load_config: OVGenAI_LoadConfig):


@app.post("/model/unload")
def unload_model():



