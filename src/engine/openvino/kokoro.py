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

        text_chunks = self.make_chunks(config.input, config.character_count_chunk)
        total_chunks = len(text_chunks)

        for idx, chunk_text in enumerate(text_chunks):

            def infer_on_chunk():
                """Blocking inference run in background thread."""
                with torch.no_grad():
                    infer = pipeline(chunk_text, voice=config.voice, speed=config.speed)
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
