import base64
import io
import json
import os
import re
import threading
import time
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

# Configuration
API_KEY = os.getenv("OPENARC_API_KEY")
BASE_URL = "http://localhost:8003/v1"
SAMPLE_RATE = 16000
# Qwen3 streaming /audio/speech uses audio/L16 mono int16 LE (see server main.py)
TTS_STREAM_SAMPLE_RATE = 24000
MODELS = {
    "asr": "qwen3_asr",
    "llm": "Muse-12B",
    # Server-registered name for a ModelType.QWEN3_TTS_VOICE_CLONE model
    "tts": os.getenv("OPENARC_QWEN3_TTS_MODEL", "voice_clone"),
}
# Qwen3 ASR config for openarc_asr.qwen3_asr (audio_base64 injected from file)
QWEN3_ASR_CONFIG = {
    "language": None,
    "max_tokens": 4096,
    "max_chunk_sec": 30.0,
    "search_expand_sec": 5.0,
    "min_window_ms": 100.0,
}
# Voice clone: reference WAV + transcript (ICL). Omit speaker (custom_voice only).
VOICE_CLONE_REF_WAV = "/home/echo/Desktop/interstellar-tars_absolute-honesty-isn-t-always-the-most-diplomatic-nor-the.mp3"
VOICE_CLONE_REF_TEXT = """
Absolute honesty isn't always the most diplomatic, nor the most tactful, nor the
"""

_ref_audio_b64_cache: Optional[str] = None


def _get_ref_audio_b64() -> str:
    """Lazy-load base64 reference WAV for qwen3_tts_voice_clone."""
    global _ref_audio_b64_cache
    if _ref_audio_b64_cache is None:
        with open(VOICE_CLONE_REF_WAV, "rb") as f:
            _ref_audio_b64_cache = base64.b64encode(f.read()).decode("ascii")
    return _ref_audio_b64_cache


# Qwen3 TTS config for openarc_tts.qwen3_tts (voice_clone mode); sampling matches OV_Qwen3TTSGenConfig
QWEN3_TTS_CONFIG = {
    "ref_text": VOICE_CLONE_REF_TEXT,
    "language": "english",
    "instruct": None,
    "x_vector_only": False,
    "max_new_tokens": 2048,
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "non_streaming_mode": True,
    "subtalker_do_sample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "stream": True,
    "stream_chunk_frames": 300,
    "stream_left_context": 25,
}
LLM_CONFIG = {
    "temperature": 0.8,
    "max_tokens": 16384,
    "top_p": 1.0,
    "repetition_penalty": 1.05
}

SYSTEM_PROMPT = """
# COMMISION: 
- You are Elmo from Sesame Street, now andventure gamemaster. 
- Make the story interactive and engaging.
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
    """Transcribe audio using Qwen3 ASR.

    Returns:
        Tuple of (transcribed_text, metrics)
    """
    response = requests.post(
        f"{BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        data={
            "model": MODELS["asr"],
            "response_format": "verbose_json",
            "openarc_asr": json.dumps({"qwen3_asr": QWEN3_ASR_CONFIG}),
        },
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


def _l16_rate_from_content_type(content_type: str) -> int:
    m = re.search(r"rate=(\d+)", content_type, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return TTS_STREAM_SAMPLE_RATE


def _play_streaming_l16(response: requests.Response, sample_rate: int) -> None:
    """Play raw little-endian int16 mono PCM as chunks arrive (Qwen3 stream)."""
    pending = bytearray()
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as out:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            pending.extend(chunk)
            n_bytes = (len(pending) // 2) * 2
            if n_bytes == 0:
                continue
            raw = bytes(pending[:n_bytes])
            del pending[:n_bytes]
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if samples.size:
                out.write(samples.reshape(-1, 1))
    if pending:
        print(f"Warning: incomplete PCM tail ({len(pending)} bytes discarded)")


def generate_and_play_speech(text: str) -> None:
    """Generate speech from text using Qwen3 TTS and play it (streams when server returns L16)."""
    url = f"{BASE_URL}/audio/speech"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    cfg = {k: v for k, v in QWEN3_TTS_CONFIG.items() if v is not None}
    cfg["input"] = text
    cfg["ref_audio_b64"] = _get_ref_audio_b64()
    data = {
        "model": MODELS["tts"],
        "input": text,
        "voice": cfg.get("speaker", MODELS["tts"]),
        "openarc_tts": {"qwen3_tts": cfg},
    }

    with requests.post(url, headers=headers, json=data, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "l16" in content_type.lower():
            print("\n🔊 Synthesizing (streaming playback)...")
            sr = _l16_rate_from_content_type(content_type)
            _play_streaming_l16(response, sr)
            print("▶️ Playback finished.")
            return

        print("\n🔊 Generating speech...")
        audio_buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                audio_buffer.write(chunk)
        audio_buffer.seek(0)
        print("▶️ Playing response...")
        audio_data, fs = sf.read(audio_buffer, dtype="float32")
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


