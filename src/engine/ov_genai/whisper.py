import asyncio
import base64
import gc
import io
import librosa
from typing import AsyncIterator, Dict, Any, Union


import numpy as np


from openvino_genai import WhisperPipeline

from src.server.model_registry import ModelRegistry, ModelLoadConfig

from src.server.models.ov_genai import OVGenAI_WhisperGenConfig


model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Whisper/distil-whisper-large-v3-int8-ov"
sample_audio_path = "/home/echo/Projects/OpenArc/src2/tests/john_steakly_armor_the_drop.wav"


class OVGenAI_Whisper:
    def __init__(self, load_config: ModelLoadConfig):
        
        self.load_config = load_config
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

    async def transcribe(self, gen_config: OVGenAI_WhisperGenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """
        Run transcription on a given base64 encoded audio and return texts with metrics.
        If `language` is provided in config, it will be used; otherwise autodetection applies.
        
        Yields in order: metrics (dict), transcribed_text (str).
        """
        # Prepare audio inputs from base64 in a worker thread
        audio_list = await asyncio.to_thread(self.prepare_audio, gen_config)

        result = await asyncio.to_thread(self.whisper_model.generate, audio_list)

        # Collect transcription and metrics
        transcription = result.texts
        perf_metrics = getattr(result, "perf_metrics", None)
        metrics_dict = self.collect_metrics(perf_metrics) if perf_metrics is not None else {}

        # transcription is the complete result from WhisperPipeline (not streaming chunks)
        final_text = transcription

        # Yield metrics first, then text (following the pattern of generate_text methods)
        yield metrics_dict
        yield final_text

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

    def load_model(self, loader: ModelLoadConfig) -> None:
        """
        Load (or reload) a Whisper model into a pipeline for the given device.
        """
        self.whisper_model = WhisperPipeline(
            loader.model_path,
            loader.device,
            **(loader.runtime_config or {})
        )

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister model from registry and free memory resources.

        Args:
            registry: ModelRegistry to unregister from
            model_id: Private model identifier returned by register_load

        Returns:
            True if the model was found and unregistered, else False.
        """
        removed = await registry.register_unload(model_name)

        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None

        gc.collect()
        print(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed

