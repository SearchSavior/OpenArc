import base64
import io
import json

import librosa
import numpy as np
import openvino as ov
from openvino_genai import WhisperGenerationConfig, WhisperPipeline

model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Whisper/distil-whisper-large-v3-int8-ov"
sample_audio_path = "/home/echo/Projects/OpenArc/src2/tests/john_steakly_armor_the_drop.wav"


@staticmethod
def audio_file_to_base64(audio_path: str) -> str:
    """
    Convert a local audio file to base64 string.
    """
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_string = base64.b64encode(audio_bytes).decode('utf-8')
        return base64_string


class OVGenAI_Whisper:
    def __init__(self, model_path: str, device: str = "GPU.1"):
        """
        Initialize an OpenVINO Whisper pipeline.
        """
        self.model_path = model_path
        self.pipeline = WhisperPipeline(model_path, device=device)

    def prepare_inputs(self, audio_base64: str) -> list[float]:
        """
        Prepare audio inputs from base64 string for the Whisper pipeline.
        """
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create a BytesIO object to simulate a file
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load audio -> float32 mono at 16kHz
        audio, sr = librosa.load(audio_buffer, sr=16000, mono=True)
        
        # Convert to list[float] as required by WhisperPipeline
        audio_list = audio.astype(np.float32).tolist()
        
        return audio_list

    def transcribe(self, audio_base64: str) -> dict:
        """
        Run transcription on a given base64 encoded audio.
        Language autodetection will be used (no language argument).
        
        Returns:
            dict: Contains transcription text and performance metrics
        """
        # Prepare audio inputs from base64
        audio_list = self.prepare_inputs(audio_base64)

        # Run transcription
        result = self.pipeline.generate(audio_list)

        # Extract performance metrics from the result
        perf_metrics = result.perf_metrics

        # Collect key performance metrics
        metrics = {
            "num_generated_tokens": perf_metrics.get_num_generated_tokens(),
            "throughput_tokens_per_sec": perf_metrics.get_throughput().mean,
            "ttft_ms": perf_metrics.get_ttft().mean,
            "load_time_ms": perf_metrics.get_load_time(),
            "generate_duration_ms": perf_metrics.get_generate_duration().mean,
            "features_extraction_duration_ms": perf_metrics.get_features_extraction_duration().mean,
            "transcription": result.texts,
            "scores": result.scores
        }

        return metrics


if __name__ == "__main__":
    whisper = OVGenAI_Whisper(model_path)
    audio_base64 = audio_file_to_base64(sample_audio_path)
    
    # Transcription with metrics
    result = whisper.transcribe(audio_base64)
    print("Transcription:")
    print(result['transcription'])
    
    # Display performance metrics as JSON
    print("\n" + "="*50)
    print("PERFORMANCE METRICS (JSON)")
    print("="*50)
    metrics = {
        "num_generated_tokens": result["num_generated_tokens"],
        "throughput_tokens_per_sec": result["throughput_tokens_per_sec"],
        "ttft_ms": result["ttft_ms"],
        "load_time_ms": result["load_time_ms"],
        "generate_duration_ms": result["generate_duration_ms"],
        "features_extraction_duration_ms": result["features_extraction_duration_ms"],
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
