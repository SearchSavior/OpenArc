# streaming_kokoro_async.py
"""
Streaming-only Kokoro + OpenVINO implementation.
Now uses asyncio.to_thread for non-blocking streaming inference.
"""

import asyncio
import gc
import json
import re

from pathlib import Path
from typing import AsyncIterator, NamedTuple

import openvino as ov
import soundfile as sf
import torch
from kokoro.model import KModel


from src.server.model_registry import ModelRegistry
from src.server.models.registration import ModelLoadConfig
from src.server.models.openvino import OV_KokoroGenConfig


class StreamChunk(NamedTuple):
    audio: torch.Tensor
    chunk_text: str
    chunk_index: int
    total_chunks: int


class OV_Kokoro(KModel):
    """
    We subclass the KModel from Kokoro to use with OpenVINO inputs.
    """
    
    def __init__(self, load_config: ModelLoadConfig):
        super().__init__()
        self.model = None
        self._device = None

    def load_model(self, load_config: ModelLoadConfig):
        self.model_path = Path(load_config.model_path)
        self._device = load_config.device

        with (self.model_path / "config.json").open("r", encoding="utf-8") as f:
            model_config = json.load(f)

        self.vocab = model_config["vocab"]
        self.context_length = model_config["plbert"]["max_position_embeddings"]

        core = ov.Core()
        self.model = core.compile_model(self.model_path / "openvino_model.xml", self._device)
        return self.model

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister model from registry and free memory resources.

        Args:
            registry: ModelRegistry to unregister from
            model_name: Model identifier to unload

        Returns:
            True if the model was found and unregistered, else False.
        """
        # Clean up model resources
        if self.model is not None:
            del self.model
            self.model = None
        
        # Unregister from registry
        removed = await registry.register_unload(model_name)
        
        # Force garbage collection to free memory
        gc.collect()
        
        return removed


    def make_chunks(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks up to `chunk_size` characters,
        preferring sentence boundaries.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # Regex: split after ., !, ? followed by space
        sentences = re.split(r'(?<=[.!?]) +', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # sentence itself longer than chunk_size -> word splitting
                    words = sentence.split()
                    temp = ""
                    for word in words:
                        if len(temp) + len(word) + 1 > chunk_size:
                            if temp:
                                chunks.append(temp.strip())
                            temp = word
                        else:
                            temp += (" " if temp else "") + word
                    current_chunk = temp
            else:
                current_chunk += (" " if current_chunk else "") + sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def chunk_forward_pass(
        self, config: OV_KokoroGenConfig
    ) -> AsyncIterator[StreamChunk]:
        """
        Async generator yielding audio chunks from text.
        Uses asyncio.to_thread to offload inference calls.
        """
        # Create pipeline with the language code from config
        from kokoro.pipeline import KPipeline
        pipeline = KPipeline(model=self, lang_code=config.lang_code.value)

        # Resolve the voice once. If voice_blend is set, this returns a
        # blended FloatTensor; otherwise the plain voice name.
        voice_arg = self._resolve_voice(config, pipeline)

        text_chunks = self.make_chunks(config.input, config.character_count_chunk)
        total_chunks = len(text_chunks)

        for idx, chunk_text in enumerate(text_chunks):

            def infer_on_chunk():
                """Blocking inference run in background thread."""
                with torch.no_grad():
                    infer = pipeline(chunk_text, voice=voice_arg, speed=config.speed)
                    result = next(infer) if hasattr(infer, "__iter__") else infer
                    return result

            # Run blocking inference off the main loop
            result = await asyncio.to_thread(infer_on_chunk)

            yield StreamChunk(
                audio=result.audio,
                chunk_text=chunk_text,
                chunk_index=idx,
                total_chunks=total_chunks,
            )

    @staticmethod
    def _parse_blend(blend: str) -> list[tuple[str, float]]:
        """Parse a blend string into [(name, weight)] with weights normalised
        to sum to 1.0. Missing weights default to 1.0, so bare comma lists
        become equal-weight averages. Names are validated upstream."""
        items: list[tuple[str, float]] = []
        for part in blend.split(","):
            part = part.strip()
            if not part:
                continue
            name, _, weight = part.partition(":")
            name = name.strip()
            try:
                w = float(weight) if weight.strip() else 1.0
            except ValueError:
                w = 1.0
            items.append((name, max(0.0, w)))
        total = sum(w for _, w in items) or 1.0
        return [(n, w / total) for n, w in items]

    def _resolve_voice(self, config: "OV_KokoroGenConfig", pipeline):
        """Return the voice argument for KPipeline. Plain voice name when
        voice_blend is unset, otherwise a weighted-sum FloatTensor of the
        named voicepacks."""
        if not getattr(config, "voice_blend", None):
            return config.voice.value if hasattr(config.voice, "value") else config.voice
        components = self._parse_blend(config.voice_blend)
        if len(components) == 1:
            return components[0][0]
        packs = [pipeline.load_single_voice(n) * w for n, w in components]
        return torch.stack(packs).sum(dim=0)
