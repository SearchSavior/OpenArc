import os
import base64
import sounddevice as sd
import soundfile as sf
import numpy as np
from pynput import keyboard
import tempfile
from openai import OpenAI
import httpx


def transcribe_example():
    """Record audio with spacebar, then transcribe using OpenAI-compatible API."""
    api_key = os.getenv("OPENARC_API_KEY")
    if not api_key:
        print("OPENARC_API_KEY is not set. Export it before running this test.")
        return

    # Initialize OpenAI client with custom HTTP client for base64 audio
    client = OpenAI(
        api_key=api_key,
        base_url="http://localhost:8000/v1"
    )
    
    model_name = "whisper"
    
    # Recording parameters
    sample_rate = 16000  # 16kHz is standard for Whisper
    recording = []
    is_recording = False
    
    def on_press(key):
        nonlocal is_recording, recording
        if key == keyboard.Key.space and not is_recording:
            is_recording = True
            recording = []
            print("\n🎤 Recording... (release spacebar to stop)")
    
    def on_release(key):
        nonlocal is_recording
        if key == keyboard.Key.space and is_recording:
            is_recording = False
            print("⏹️  Recording stopped. Processing...")
            return False  # Stop listener
        elif key == keyboard.Key.esc:
            print("\n❌ Cancelled")
            return False
    
    print("Press and hold SPACEBAR to record audio")
    print("Press ESC to cancel")
    
    # Audio callback - captures audio while spacebar is held
    def audio_callback(indata, frames, time, status):
        if is_recording:
            recording.append(indata.copy())
    
    # Start audio stream
    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
    stream.start()
    
    # Listen for keyboard events
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    
    stream.stop()
    stream.close()
    
    if not recording:
        print("No audio recorded")
        return
    
    # Convert captured audio to numpy array
    audio_data = np.concatenate(recording, axis=0)
    
    # Save recorded audio as WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        sf.write(tmp_path, audio_data, sample_rate)
    
    print(f"💾 Audio saved to temporary WAV file")
    
    try:
        # Read WAV file and encode as base64 (OpenArc server expects this format)
        with open(tmp_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Use custom request since OpenArc expects audio_base64 in JSON
        response = client.post(
            "/audio/transcriptions",
            cast_to=object,
            body={
                "model": model_name,
                "audio_base64": audio_b64
            },
            options={"headers": {"Content-Type": "application/json"}}
        )
        
        text = response.get("text", "")
        metrics = response.get("metrics", {})
        
        print("\n📝 Transcription:\n", text)
        
        if metrics:
            print("\nMetrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
                
    except Exception as e:
        print(f"Transcription failed: {e}")
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


if __name__ == "__main__":
    transcribe_example()


