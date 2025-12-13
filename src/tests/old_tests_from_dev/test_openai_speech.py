import os
from pathlib import Path
import requests

api_key = os.getenv("OPENARC_API_KEY")

url = "http://localhost:8000/v1/audio/speech"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "kokoro",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "af_sarah",
    "speed": 1.0,
    "language": "a",
    "response_format": "wav"
}

speech_file_path = Path(__file__).parent / "speech2.wav"

with requests.post(url, headers=headers, json=data, stream=True) as response:
    response.raise_for_status()
    with open(speech_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
