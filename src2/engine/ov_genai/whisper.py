import asyncio
import base64
import gc
import io
import json

from pydantic import BaseModel, Field

import librosa
import numpy as np
from openvino_genai import WhisperGenerationConfig, WhisperPipeline

model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Whisper/distil-whisper-large-v3-int8-ov"
sample_audio_path = "/home/echo/Projects/OpenArc/src2/tests/john_steakly_armor_the_drop.wav"



class OVGenAI_WhisperGenConfig(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio")


def audio_file_to_base64(audio_path: str) -> str:
    """
    Convert a local audio file to base64 string.
    """
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_string = base64.b64encode(audio_bytes).decode('utf-8')
        return base64_string

class OVGenAI_Whisper:
    def __init__(self):
        """
        Do not initialize or store model/pipeline state here.
        """
        pass

    def prepare_audio(self, gen_config: OVGenAI_WhisperGenConfig) -> list[float]:
        """
        Prepare audio inputs from base64 string for the Whisper pipeline.
        """
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(gen_config.audio_base64)
        
        # Create a BytesIO object to simulate a file
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load audio -> float32 mono at 16kHz
        audio, sr = librosa.load(audio_buffer, sr=16000, mono=True)
        # Return as a Python list[float] (float32 -> float) for pybind compatibility
        return audio.astype(np.float32).tolist()

    async def transcribe(self, gen_config: OVGenAI_WhisperGenConfig) -> list[str]:
        """
        Run transcription on a given base64 encoded audio.
        If `language` is provided in config, it will be used; otherwise autodetection applies.
        
        Returns:
            list[str]: transcription_texts
        """
        # Prepare audio inputs from base64 in a worker thread
        audio_list = await asyncio.to_thread(self.prepare_audio, gen_config)


        result = await asyncio.to_thread(self.whisper_model.generate, audio_list)

        # Collect transcription
        transcription = result.texts

        return transcription

    def collect_metrics(self, perf_metrics) -> dict:
        """
        Collect key performance metrics from a Whisper perf_metrics object.
        """
        metrics = {
            "num_generated_tokens": perf_metrics.get_num_generated_tokens(),
            "throughput_tokens_per_sec": round(perf_metrics.get_throughput().mean, 4),
            "ttft_s": round(perf_metrics.get_ttft().mean / 1000, 4),
            "load_time_s": round(perf_metrics.get_load_time() / 1000, 4),
            "generate_duration_s": round(perf_metrics.get_generate_duration().mean / 1000, 4),
            "features_extraction_duration_ms": round(perf_metrics.get_features_extraction_duration().mean, 4),
        }

        return metrics

    def load_model(self, model_path: str, device: str = "GPU.1") -> None:
        """
        Load (or reload) a Whisper model into a pipeline for the given device.
        """
        self.whisper_model = WhisperPipeline(
            model_path, 
            device=device
            
            )

    def unload_model(self) -> None:
        """
        Unload the currently loaded Whisper pipeline and free resources.
        """
        if hasattr(self, "whisper_model"):
            del self.whisper_model
            gc.collect()

if __name__ == "__main__":
    async def _demo():
        whisper = OVGenAI_Whisper()
        whisper.load_model(model_path)
        audio_base64 = audio_file_to_base64(sample_audio_path)

        # Transcription with metrics
        gen_config = OVGenAI_WhisperGenConfig(audio_base64=audio_base64)
        transcription2 = await whisper.transcribe(gen_config)
        audio_list = whisper.prepare_audio(gen_config)
        metrics2 = whisper.collect_metrics(whisper.whisper_model.generate(audio_list).perf_metrics)
        print("Transcription:")
        print(transcription2)

        # Display performance metrics as JSON
        print("\n" + "="*50)
        print("PERFORMANCE METRICS (JSON)")
        print("="*50)
        print(json.dumps(metrics2, indent=2, ensure_ascii=False))

    asyncio.run(_demo())
