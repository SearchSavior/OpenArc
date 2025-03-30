import os
import gradio as gr

class QueryAPI:
    def __init__(self):
        self.api_key = os.environ.get("OPENARC_API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

class ModelManager:
    def __init__(self):
        self.models = []
        self.model_ids = []