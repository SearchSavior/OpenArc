import io
import os
import threading
import time
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

# Configuration
API_KEY = os.getenv("OPENARC_API_KEY")
BASE_URL = "http://localhost:8000/v1"
SAMPLE_RATE = 16000
MODELS = {
    "whisper": "whisper",
    "llm": "Muse-12B",
    "tts": "kokoro"
}
TTS_CONFIG = {
    "voice": "af_heart",
    "speed": 1.25,
    "language": "a",
    "response_format": "wav"
}
LLM_CONFIG = {
    "temperature": 0.8,
    "max_tokens": 16384,
    "top_p": 1.0,
    "repetition_penalty": 1.05
}

SYSTEM_PROMPT = """
# COMMISION: 
- You're a masterful adventure gamemaster. 
- ALways make the story interactive, and dont tell to much.
- Use second person (you are)

## STYLE
Any time the user harms a character it should be very violent; executions are always brutal.

## Player
My name is Leandro.
"""

def validate_api_key() -> None:
    """Validate required API key for OpenArc calls."""
    if not API_KEY:
        raise RuntimeError("OPENARC_API_KEY environment variable not set")

def record_audio() -> tuple[Optional[np.ndarray], bool]:
    """Record audio from microphone using terminal input (works in remote desktop).
    
    Press ENTER to start recording, press ENTER again to stop.
    Type 'q' + ENTER to quit.
    
    Returns:
        Tuple of (audio_data, exit_program)
    """
    recording = []
    recording_lock = threading.Lock()
    is_recording = threading.Event()
    exit_program = False
    recording_stopped = threading.Event()
    
    print("\n" + "="*60)
    print("  🎤 Audio Recording Control")
    print("="*60)
    print("  [ENTER] - Start/Stop recording")
    print("  [q + ENTER] - Quit conversation")
    print("="*60)
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        if is_recording.is_set():
            with recording_lock:
                recording.append(indata.copy())
    
    def input_thread():
        """Thread that waits for user input."""
        nonlocal exit_program, recording_stopped
        
        while not recording_stopped.is_set():
            try:
                if not is_recording.is_set():
                    # Waiting to start recording
                    print("\nPress ENTER to start recording (or 'q' + ENTER to quit)...")
                else:
                    # Recording in progress
                    print("\n🎤 Recording... Press ENTER to stop...")
                
                user_text = input().strip().lower()
                
                if user_text == 'q':
                    print("\n❌ Exiting conversation")
                    exit_program = True
                    is_recording.clear()
                    recording_stopped.set()
                    break
                elif not is_recording.is_set():
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
                exit_program = True
                is_recording.clear()
                recording_stopped.set()
                break
    
    # Start audio stream
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
        stream.start()
        # Give stream a moment to initialize
        time.sleep(0.1)
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        print("Available audio devices:")
        print(sd.query_devices())
        return None, False
    
    # Start input thread
    input_handler = threading.Thread(target=input_thread, daemon=True)
    input_handler.start()
    
    try:
        # Wait for recording to stop or exit
        recording_stopped.wait()
    finally:
        stream.stop()
        stream.close()
    
    # Convert to numpy array if audio was recorded
    with recording_lock:
        recording_copy = recording.copy()
    
    if exit_program:
        return None, exit_program
    
    if not recording_copy or len(recording_copy) == 0:
        print("Warning: No audio data was recorded")
        return None, exit_program
    
    print(f"Recorded {len(recording_copy)} audio chunks")
    audio_data = np.concatenate(recording_copy, axis=0)
    print(f"Total audio length: {len(audio_data) / SAMPLE_RATE:.2f} seconds")
    return audio_data, exit_program

def encode_audio_to_wav_bytes(audio_data: np.ndarray) -> bytes:
    """Convert audio data to in-memory WAV bytes."""
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_data, SAMPLE_RATE, format='WAV')
    wav_buffer.seek(0)
    return wav_buffer.read()

def transcribe_audio(audio_bytes: bytes) -> tuple[str, dict]:
    """Transcribe audio using Whisper model.
    
    Returns:
        Tuple of (transcribed_text, metrics)
    """
    response = requests.post(
        f"{BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        data={"model": MODELS["whisper"]},
        files={"file": ("recording.wav", audio_bytes, "audio/wav")},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("text", "").strip()
    metrics = payload.get("metrics", {})
    return text, metrics

def get_llm_response(messages: list[dict]) -> str:
    """Get response from LLM."""
    print("\n🤖 Thinking...")
    
    # Use direct HTTP request to ensure custom parameters (repetition_penalty, top_k) are sent correctly
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODELS["llm"],
        "messages": messages,
        "stream": False,
        **LLM_CONFIG
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    
    full_response = result["choices"][0]["message"]["content"]
    print(f"\nLLM Response:\n{full_response}\n")
    
    return full_response

def generate_and_play_speech(text: str) -> None:
    """Generate speech from text using TTS and play it."""
    print("\n🔊 Generating speech...")
    url = f"{BASE_URL}/audio/speech"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODELS["tts"],
        "input": text,
        **TTS_CONFIG
    }
    
    audio_buffer = io.BytesIO()
    with requests.post(url, headers=headers, json=data, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                audio_buffer.write(chunk)
    
    audio_buffer.seek(0)
    
    # Play audio from memory
    print("▶️ Playing response...")
    audio_data, fs = sf.read(audio_buffer, dtype='float32')
    sd.play(audio_data, fs)
    sd.wait()

def talk_to_llm():
    """Maintain a conversation: record -> transcribe -> LLM -> TTS -> repeat."""
    try:
        validate_api_key()
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    # Initialize conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    while True:
        # Record audio
        audio_data, exit_program = record_audio()
        
        if exit_program:
            break
        
        if audio_data is None:
            print("No audio recorded, skipping...")
            continue
        
        try:
            # Transcribe audio
            audio_bytes = encode_audio_to_wav_bytes(audio_data)
            text, metrics = transcribe_audio(audio_bytes)
            
            if not text:
                print("No transcription, skipping...")
                continue
            
            print(f"\n📝 You said:\n{text}")
            
            if metrics:
                print("\nTranscription Metrics:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
            
            # Get LLM response
            messages.append({"role": "user", "content": text})
            llm_response = get_llm_response(messages)
            
            if llm_response.strip():
                messages.append({"role": "assistant", "content": llm_response})
                
                # Generate and play TTS
                generate_and_play_speech(llm_response)
        
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    talk_to_llm()


