import os
import base64
import requests


def transcribe_example():
    """Transcribe a local WAV file using OpenAI-compatible /v1/audio/transcriptions."""
    api_key = os.getenv("OPENARC_API_KEY")
    if not api_key:
        print("OPENARC_API_KEY is not set. Export it before running this test.")
        return

    model_name = "whisper"
    audio_path = "/home/echo/Projects/OpenArc/src/tests/john_steakly_armor_the_drop.wav"

    try:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Failed to read audio file: {e}")
        return

    url = "http://localhost:8000/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "audio_base64": audio_b64,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        print("Status:", resp.status_code)
        if resp.status_code != 200:
            print("Error:", resp.text)
            return
        data = resp.json()
        text = data.get("text", "")
        metrics = data.get("metrics", {})
        print("Transcription:\n", text)
        if metrics:
            print("\nMetrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    transcribe_example()


