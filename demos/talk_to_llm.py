import base64
import io  # For in-memory audio handling
import os

import numpy as np
import requests  # For TTS request
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from pynput import keyboard


def talk_to_llm():
    """Maintain a conversation: record -> transcribe -> LLM -> TTS -> repeat."""
    api_key = os.getenv("OPENARC_API_KEY")
    if not api_key:
        print("OPENARC_API_KEY is not set. Export it before running this test.")
        return

    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url="http://localhost:8000/v1"
    )
    
    # Conversation history
    messages = [
        {
            "role": "system",
            "content": "You are an adventre game master who tels interactive stories"
        }
    ]
    
    whisper_model = "whisper"
    llm_model = "Dolphin-X1"
    tts_model = "kokoro"
    sample_rate = 16000  # For recording
    
    while True:
        # Recording setup
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
                return False  # Stop listener
            elif key == keyboard.Key.esc:
                print("\n❌ Exiting conversation")
                exit_program = True
                return False
        
        print("\nPress and hold SPACEBAR to record your message (ESC to exit)")
        
        # Audio callback
        def audio_callback(indata, frames, time, status):
            if is_recording:
                recording.append(indata.copy())
        
        # Start audio stream
        stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
        stream.start()
        
        # Listen for keyboard
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
        
        if exit_program:
            break
        
        stream.stop()
        stream.close()
        
        if not recording:
            print("No audio recorded, skipping...")
            continue
        
        # Convert to numpy array
        audio_data = np.concatenate(recording, axis=0)
        
        # Create in-memory WAV
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
        wav_buffer.seek(0)
        audio_bytes = wav_buffer.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        try:
            # Transcribe
            response = client.post(
                "/audio/transcriptions",
                cast_to=object,
                body={
                    "model": whisper_model,
                    "audio_base64": audio_b64
                },
                options={"headers": {"Content-Type": "application/json"}}
            )
            
            text = response.get("text", "").strip()
            metrics = response.get("metrics", {})
            
            if text:
                print("\n📝 You said:\n", text)
                messages.append({"role": "user", "content": text})
            else:
                print("No transcription, skipping...")
                continue
            
            if metrics:
                print("\nTranscription Metrics:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
            
            # Get LLM response
            print("\n🤖 Thinking...")
            stream_response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                stream=True
            )
            
            print("\nLLM Response: ", end="")
            full_response = ""
            for chunk in stream_response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    print(content, end="", flush=True)
                    full_response += content
            print("\n")
            
            if full_response.strip():
                messages.append({"role": "assistant", "content": full_response})
                
                # Generate TTS from latest assistant message
                print("\n🔊 Generating speech...")
                url = "http://localhost:8000/v1/audio/speech"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": tts_model,
                    "input": full_response,
                    "voice": "af_sarah",
                    "speed": 1.0,
                    "language": "a",
                    "response_format": "wav"
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
            
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    talk_to_llm()


