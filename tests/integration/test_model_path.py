import os
from pathlib import Path

DEFAULT_TEST_MODEL_PATH = r"/mnt/Ironwolf-4TB/Models/"

def model_path(model_name: str) -> Path:
    if not os.getenv("TEST_MODEL_PATH"):
        print("set TEST_MODEL_PATH environment variable to override default model path")
    root = os.getenv("TEST_MODEL_PATH", DEFAULT_TEST_MODEL_PATH)
    return Path(root) / model_name

