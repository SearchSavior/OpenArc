import base64
import io
import os
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from pynput import keyboard


# Configuration
API_KEY = os.getenv("OPENARC_API_KEY")
BASE_URL = "http://localhost:8000/v1"
SAMPLE_RATE = 16000
MODELS = {
    "whisper": "whisper",
    "llm": "Cydonia-24B",
    "tts": "kokoro"
}
TTS_CONFIG = {
    "voice": "af_sarah",
    "speed": 1.25,
    "language": "a",
    "response_format": "wav"
}
# SYSTEM_PROMPT = "You are a text adventure model. Your job is to keep the story moving forward in a natural way. Use the second person when referring to the users actions. Try not to present choices- instead, describe the situation, and use language to communicate where the story has potential to go."
SYSTEM_PROMPT = "You are a text adventure game narrator. Write all narration in second person (you). Keep responses brief (2-4 paragraphs max) and always end with a vivid situation that invites action. Never stall—always introduce new developments, encounters, or complications. Describe what the player sees, hears, and can interact with. When the player acts, show immediate consequences and move the story forward. Maintain consistent world rules and remember previous events. Let players decide their own actions—describe the scene, not the options."

def initialize_client() -> OpenAI:
    """Initialize OpenAI client with OpenArc server."""
    if not API_KEY:
        raise RuntimeError("OPENARC_API_KEY environment variable not set")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def record_audio() -> tuple[Optional[np.ndarray], bool]:
    """Record audio from microphone until spacebar is released or ESC is pressed.
    
    Returns:
        Tuple of (audio_data, exit_program)
    """
    recording = []
    is_recording = False
    exit_program = False
    
    def on_press(key):
        nonlocal is_recording, recording
        if key == keyboard.Key.space and not is_recording:
            is_recording = True
            recording = []
            print("\n🎤 Recording... (release spacebar to stop)")
    
    def on_release(key):
        nonlocal is_recording, exit_program
        if key == keyboard.Key.space and is_recording:
            is_recording = False
            print("⏹️  Recording stopped. Processing...")
            return False
        elif key == keyboard.Key.esc:
            print("\n❌ Exiting conversation")
            exit_program = True
            return False
    
    print("\nPress and hold SPACEBAR to record your message (ESC to exit)")
    
    def audio_callback(indata, frames, time, status):
        if is_recording:
            recording.append(indata.copy())
    
    # Record audio
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
    stream.start()
    
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    
    stream.stop()
    stream.close()
    
    # Convert to numpy array if audio was recorded
    if not recording:
        return None, exit_program
    
    audio_data = np.concatenate(recording, axis=0)
    return audio_data, exit_program

def encode_audio_to_base64(audio_data: np.ndarray) -> str:
    """Convert audio data to base64-encoded WAV format."""
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_data, SAMPLE_RATE, format='WAV')
    wav_buffer.seek(0)
    audio_bytes = wav_buffer.read()
    return base64.b64encode(audio_bytes).decode("utf-8")

def transcribe_audio(client: OpenAI, audio_b64: str) -> tuple[str, dict]:
    """Transcribe audio using Whisper model.
    
    Returns:
        Tuple of (transcribed_text, metrics)
    """
    response = client.post(
        "/audio/transcriptions",
        cast_to=object,
        body={
            "model": MODELS["whisper"],
            "audio_base64": audio_b64
        },
        options={"headers": {"Content-Type": "application/json"}}
    )
    
    text = response.get("text", "").strip()
    metrics = response.get("metrics", {})
    return text, metrics

def get_llm_response(client: OpenAI, messages: list[dict]) -> str:
    """Get response from LLM."""
    print("\n🤖 Thinking...")
    response = client.chat.completions.create(
        model=MODELS["llm"],
        messages=messages,
        stream=False,
        max_tokens=16384
    )
    
    full_response = response.choices[0].message.content
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
        client = initialize_client()
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
            audio_b64 = encode_audio_to_base64(audio_data)
            text, metrics = transcribe_audio(client, audio_b64)
            
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
            llm_response = get_llm_response(client, messages)
            
            if llm_response.strip():
                messages.append({"role": "assistant", "content": llm_response})
                
                # Generate and play TTS
                generate_and_play_speech(llm_response)
        
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    talk_to_llm()


