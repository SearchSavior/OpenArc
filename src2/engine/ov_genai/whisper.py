import base64
import gc
import io
import json

import librosa
import numpy as np
from openvino_genai import WhisperPipeline

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
    def __init__(self):
        """
        Do not initialize or store model/pipeline state here.
        """
        pass


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

    def transcribe(self, audio_base64: str) -> list[str]:
        """
        Run transcription on a given base64 encoded audio.
        Language autodetection will be used (no language argument).
        
        Returns:
            list[str]: transcription_texts
        """
        # Prepare audio inputs from base64
        audio_list = self.prepare_inputs(audio_base64)

        # Run transcription
        result = self.pipeline.generate(audio_list)

        # Collect metrics and transcription separately
        transcription = result.texts

        return transcription

    def load_model(self, model_path: str, device: str = "GPU.1") -> None:
        """
        Load (or reload) a Whisper model into a pipeline for the given device.
        """
        self.pipeline = WhisperPipeline(
            model_path, 
            device=device
            )

    def unload_model(self) -> None:
        """
        Unload the currently loaded Whisper pipeline and free resources.
        """
        if hasattr(self, "pipeline"):
            del self.pipeline
            gc.collect()

if __name__ == "__main__":
    whisper = OVGenAI_Whisper()
    whisper.load_model(model_path)
    audio_base64 = audio_file_to_base64(sample_audio_path)
    
    # Transcription with metrics
    transcription1 = whisper.transcribe(audio_base64)
    audio_list = whisper.prepare_inputs(audio_base64)
    metrics1 = whisper.collect_metrics(whisper.pipeline.generate(audio_list).perf_metrics)
    transcription2 = whisper.transcribe(audio_base64)
    audio_list = whisper.prepare_inputs(audio_base64)
    metrics2 = whisper.collect_metrics(whisper.pipeline.generate(audio_list).perf_metrics)
    print("Transcription:")
    print(transcription2)
    
    # Display performance metrics as JSON
    print("\n" + "="*50)
    print("PERFORMANCE METRICS (JSON)")
    print("="*50)
    print(json.dumps(metrics2, indent=2, ensure_ascii=False))
