import os
import base64
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import tempfile
from openai import OpenAI


def transcribe_example():
    """Record audio with ENTER key, then transcribe using OpenAI-compatible API."""
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
    recording_lock = threading.Lock()
    is_recording = threading.Event()
    recording_stopped = threading.Event()
    
    print("\n" + "="*60)
    print("  🎤 Audio Recording Control")
    print("="*60)
    print("  [ENTER] - Start/Stop recording")
    print("="*60)
    
    # Audio callback - captures audio while recording
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        if is_recording.is_set():
            with recording_lock:
                recording.append(indata.copy())
    
    def input_thread():
        """Thread that waits for user input."""
        nonlocal recording_stopped
        
        while not recording_stopped.is_set():
            try:
                if not is_recording.is_set():
                    # Waiting to start recording
                    print("\nPress ENTER to start recording...")
                else:
                    # Recording in progress
                    print("\n🎤 Recording... Press ENTER to stop...")
                
                input()  # Wait for ENTER key
                
                if not is_recording.is_set():
                    # Start recording
                    with recording_lock:
                        recording.clear()
                    is_recording.set()
                    print("\n🎤 Recording started!")
                else:
                    # Stop recording
                    is_recording.clear()
                    recording_stopped.set()
                    print("\n⏹️  Recording stopped. Processing...")
                    break
            except (EOFError, KeyboardInterrupt):
                is_recording.clear()
                recording_stopped.set()
                break
    
    # Start audio stream
    try:
        stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
        stream.start()
        # Give stream a moment to initialize
        time.sleep(0.1)
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    # Start input thread
    input_handler = threading.Thread(target=input_thread, daemon=True)
    input_handler.start()
    
    try:
        # Wait for recording to stop
        recording_stopped.wait()
    finally:
        stream.stop()
        stream.close()
    
    # Convert to numpy array if audio was recorded
    with recording_lock:
        recording_copy = recording.copy()
    
    if not recording_copy or len(recording_copy) == 0:
        print("No audio recorded")
        return
    
    # Convert captured audio to numpy array
    audio_data = np.concatenate(recording_copy, axis=0)
    
    # Save recorded audio as WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        sf.write(tmp_path, audio_data, sample_rate)
    
    print("💾 Audio saved to temporary WAV file")
    
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


