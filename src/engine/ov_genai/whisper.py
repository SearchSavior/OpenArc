import asyncio
import base64
import gc
import io
import logging
import os
import subprocess
import tempfile
from typing import Any, AsyncIterator, Dict, Union

import librosa
import numpy as np
from openvino_genai import WhisperPipeline
from soundfile import LibsndfileError

from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import ModelLoadConfig
from src.server.schemas.modeling.contract_whisper import OVGenAI_WhisperGenConfig

logger = logging.getLogger(__name__)


class OVGenAI_Whisper:
    def __init__(self, load_config: ModelLoadConfig):
        
        self.load_config = load_config
        pass

    def decode_with_ffmpeg(self, audio_bytes: bytes) -> np.ndarray:
        src = tempfile.NamedTemporaryFile(delete=False)
        try:
            src.write(audio_bytes)
            src.flush()
            src.close()
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel", "error",
                    "-i", src.name,         # Read the uploaded audio from the temporary file.
                    "-ac", "1",             # Convert the audio to mono.
                    "-ar", "16000",         # Resample to 16 kHz.
                    "-f", "f32le",          # Emit raw 32-bit little-endian float PCM.
                    "pipe:1",               # Write the decoded audio to std out.
                ],
                capture_output=True,
            )
            if proc.returncode != 0:
                raise ValueError(f"ffmpeg could not decode audio: {proc.stderr.decode(errors='replace')[:500]}")
            return np.frombuffer(proc.stdout, dtype=np.float32)
        finally:
            os.unlink(src.name)  # always delete the temp file

    def prepare_audio(self, gen_config: OVGenAI_WhisperGenConfig) -> list[float]:
        """
        Prepare audio inputs from base64 string for the Whisper pipeline.
        """

        audio_bytes = base64.b64decode(gen_config.audio_base64)
        
        audio_buffer = io.BytesIO(audio_bytes)

        try:
            audio, sr = librosa.load(audio_buffer, sr=16000, mono=True)
        except LibsndfileError as exc:
            logger.info(f"librosa decode failed ({exc}), retrying with ffmpeg")
            audio = self.decode_with_ffmpeg(audio_bytes)

        return audio.astype(np.float32).tolist()

    async def transcribe(self, gen_config: OVGenAI_WhisperGenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """
        Run transcription on a given base64 encoded audio and return texts with metrics.
        
        Yields in order: metrics (dict), transcribed_text (str).
        """
        audio_list = await asyncio.to_thread(self.prepare_audio, gen_config)

        result = await asyncio.to_thread(self.whisper_model.generate, audio_list)

        # Collect transcription and metrics
        transcription = result.texts
        perf_metrics = getattr(result, "perf_metrics", None)
        metrics_dict = self.collect_metrics(perf_metrics) if perf_metrics is not None else {}

        final_text = ' '.join(transcription) if isinstance(transcription, list) else transcription

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
        pipeline_kwargs = {**(loader.runtime_config or {})}
        if loader.cache_dir:
            pipeline_kwargs['CACHE_DIR'] = loader.cache_dir

        self.whisper_model = WhisperPipeline(
            loader.model_path,
            loader.device,
            **pipeline_kwargs
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
        logger.info(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed

