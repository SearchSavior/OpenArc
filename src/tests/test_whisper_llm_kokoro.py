#!/usr/bin/env python3
"""
Comprehensive test that chains Whisper transcription -> LLM chat completion -> Kokoro TTS
"""

import os
import base64
import requests
import json
import time
from pathlib import Path
from openai import OpenAI


def load_models():
    """Load all required models: Whisper to GPU.2, LLM to GPU.1, Kokoro to CPU"""
    api_key = os.getenv("OPENARC_API_KEY")
    if not api_key:
        print("OPENARC_API_KEY is not set. Export it before running this test.")
        return False

    base_url = "http://localhost:8000/openarc/load"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Model configurations
    models_config = [
        {
            "model_path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Whisper/distil-whisper-large-v3-int8-ov",
            "model_name": "distil-whisper-large-v3-int8-ov",
            "model_type": "whisper",
            "engine": "ovgenai",
            "device": "GPU.2",
            "runtime_config": {}
        },
        {
            "model_path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Mistral/Rocinante-12B-v1.1-int4_sym-awq-se-ov",
            "model_name": "Hermes-4-14B-int4_sym-ov",
            "model_type": "llm",
            "engine": "ovgenai",
            "device": "GPU.1",
            "runtime_config": {}
        },
        {
            "model_path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Kokoro-82M-FP16-OpenVINO",
            "model_name": "kokoro-82m-fp16-ov",
            "model_type": "kokoro",
            "engine": "openvino",
            "device": "CPU",
            "runtime_config": {}
        }
    ]

    print("Loading models...")
    for i, config in enumerate(models_config):
        print(f"Loading model {i+1}/3: {config['model_name']} on {config['device']}")
        try:
            response = requests.post(base_url, headers=headers, json=config, timeout=300)
            if response.status_code == 200:
                print(f"✓ Successfully loaded {config['model_name']}")
            else:
                print(f"✗ Failed to load {config['model_name']}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"✗ Error loading {config['model_name']}: {e}")
            return False
        
        # Wait longer for LLM model (index 1)
        if i == 1:  # LLM model
            print("⏳ Waiting 60 seconds for LLM model to fully initialize...")
            time.sleep(60)
        else:
            # Brief pause between other model loads
            time.sleep(5)

    print("All models loaded successfully!")
    return True


def verify_models():
    """Verify all models are loaded and available via /v1/models endpoint"""
    api_key = os.getenv("OPENARC_API_KEY")
    
    expected_models = [
        "distil-whisper-large-v3-int8-ov",
        "Hermes-4-14B-int4_sym-ov", 
        "kokoro-82m-fp16-ov"
    ]
    
    print("Verifying models via /v1/models endpoint...")
    
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=api_key,
        )
        
        response = client.models.list()
        loaded_models = [model.id for model in response.data]
        
        print(f"Found {len(loaded_models)} loaded models:")
        for model in sorted(loaded_models):
            print(f"  - {model}")
        
        print("\nVerification results:")
        all_loaded = True
        for expected in expected_models:
            if expected in loaded_models:
                print(f"✓ {expected} - LOADED")
            else:
                print(f"✗ {expected} - MISSING")
                all_loaded = False
        
        if all_loaded:
            print("🎉 All required models are loaded and ready!")
            return True
        else:
            print("❌ Some models are missing. Check the loading process.")
            return False
            
    except Exception as e:
        print(f"Error verifying models: {e}")
        return False


def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper"""
    api_key = os.getenv("OPENARC_API_KEY")
    
    print(f"Transcribing audio file: {audio_path}")
    
    try:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Failed to read audio file: {e}")
        return None

    url = "http://localhost:8000/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "distil-whisper-large-v3-int8-ov",
        "audio_base64": audio_b64,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            print(f"Transcription failed: {resp.status_code} - {resp.text}")
            return None
        
        data = resp.json()
        text = data.get("text", "")
        metrics = data.get("metrics", {})
        
        print("✓ Transcription completed:")
        print(f"Text: {text}")
        if metrics:
            print("Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        
        return text
        
    except Exception as e:
        print(f"Transcription request failed: {e}")
        return None


def chat_completion(transcribed_text):
    """Send transcribed text to LLM for chat completion"""
    api_key = os.getenv("OPENARC_API_KEY")
    
    print("Generating chat completion...")
    
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=api_key,
        )
        
        # Create a more interesting prompt based on the transcribed text
        system_prompt = "You are a helpful assistant. Respond thoughtfully and conversationally to the user's input."
        user_prompt = f"{transcribed_text}"
        
        resp = client.chat.completions.create(
            model="Hermes-4-14B-int4_sym-ov",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.7
        )
        
        if resp and resp.choices:
            completion_text = resp.choices[0].message.content
            print("✓ Chat completion generated:")
            print(f"Response: {completion_text}")
            return completion_text
        else:
            print("No completion generated")
            return None
            
    except Exception as e:
        print(f"Chat completion error: {e}")
        return None


def text_to_speech(text, output_path):
    """Convert text to speech using Kokoro TTS"""
    api_key = os.getenv("OPENARC_API_KEY")
    
    print("Converting text to speech...")
    
    url = "http://localhost:8000/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "kokoro-82m-fp16-ov",
        "input": text,
        "voice": "af_sarah",
        "speed": 1.0,
        "language": "a",
        "response_format": "wav"
    }
    
    try:
        with requests.post(url, headers=headers, json=data, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        print(f"✓ Speech synthesis completed. Audio saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        return False


def main():
    """Main test function that chains all operations"""
    print("=== OpenArc Whisper -> LLM -> Kokoro Chain Test ===\n")
    
    # Configuration
    sample_input_audio = "/home/echo/Projects/OpenArc/src/tests/litany_against_fear_dune.wav"
    output_audio_path = Path(__file__).parent / "whisper_llm_kokoro_output.wav"
    
    # Step 1: Load all models
    print("Step 1: Loading models...")
    if not load_models():
        print("Failed to load models. Exiting.")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Step 2: Verify all models are loaded
    print("Step 2: Verifying models...")
    if not verify_models():
        print("Model verification failed. Exiting.")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Step 3: Transcribe audio
    print("Step 3: Transcribing audio...")
    transcribed_text = transcribe_audio(sample_input_audio)
    if not transcribed_text:
        print("Transcription failed. Exiting.")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Step 4: Generate chat completion
    print("Step 4: Generating chat completion...")
    completion_text = chat_completion(transcribed_text)
    if not completion_text:
        print("Chat completion failed. Exiting.")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Step 5: Convert to speech
    print("Step 5: Converting to speech...")
    if not text_to_speech(completion_text, output_audio_path):
        print("Text-to-speech failed. Exiting.")
        return
    
    print("\n" + "="*60)
    print("🎉 SUCCESS! Complete pipeline executed:")
    print(f"   Input audio: {sample_input_audio}")
    print(f"   Transcribed: '{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}'")
    print(f"   LLM response: '{completion_text[:100]}{'...' if len(completion_text) > 100 else ''}'")
    print(f"   Output audio: {output_audio_path}")
    print("="*60)


if __name__ == "__main__":
    main()
