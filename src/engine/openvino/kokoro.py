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
from src.server.schemas.registration import ModelLoadConfig
from src.server.schemas.modeling.contract_kokoro import OV_KokoroGenConfig


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
        if load_config.cache_dir:
            core.set_property({"CACHE_DIR": load_config.cache_dir})
        if load_config.runtime_config:
            core.set_property(load_config.runtime_config)
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


    # Sentence end: .!? (optionally followed by quotes/brackets), then whitespace.
    _SENTENCE_RE = re.compile(r'(?<=[.!?…])["\')\]]*\s+')
    # Clause punctuation worth pausing on when a mid-sentence split is needed.
    _CLAUSE_PUNCT = ",;:\u2014\u2013-"
    # Prefer clause splits at least this far into the chunk to avoid
    # degenerate tiny heads (e.g. a comma at position 4).
    _MIN_CLAUSE_FRACTION = 0.3

    @staticmethod
    def _split_head(text: str, limit: int) -> tuple[str, str]:
        """Split off a head of at most `limit` chars, preferring a clause
        boundary, then a word boundary, then a hard cut. Keeps punctuation
        on the head so the model still hears the pause."""
        head = text[:limit]
        cut = -1
        best = max(head.rfind(p) for p in OV_Kokoro._CLAUSE_PUNCT)
        if best >= int(limit * OV_Kokoro._MIN_CLAUSE_FRACTION):
            cut = best + 1
        else:
            cut = head.rfind(" ")
            if cut <= 0:
                cut = limit
        return text[:cut].strip(), text[cut:].strip()

    def make_chunks(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks of at most `chunk_size` characters.

        Boundary preference: paragraph (blank line) -> sentence -> clause
        (comma/semicolon/colon/dash) -> word -> hard cut. Guaranteed: every
        returned chunk respects the size limit and no character is ever
        dropped (concatenation reproduces the input modulo whitespace).
        """
        if not text or not text.strip():
            return []
        if len(text.strip()) <= chunk_size:
            return [text.strip()]

        # Paragraph breaks are hard boundaries; sentences within paragraphs.
        segments: list[str] = []
        for paragraph in re.split(r'\n\s*\n+|\n', text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            segments.extend(s for s in self._SENTENCE_RE.split(paragraph) if s.strip())

        chunks: list[str] = []
        current = ""

        for seg in segments:
            if len(seg) > chunk_size:
                # Oversized segment: flush buffer, then carve off heads until
                # the remainder fits. The tail becomes the new buffer so it
                # can merge with the next sentence.
                if current:
                    chunks.append(current)
                    current = ""
                while len(seg) > chunk_size:
                    head, seg = self._split_head(seg, chunk_size)
                    chunks.append(head)
            if not current:
                current = seg
            elif len(current) + 1 + len(seg) <= chunk_size:
                current = f"{current} {seg}"
            else:
                chunks.append(current)
                current = seg

        if current:
            chunks.append(current)

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
                    if not hasattr(infer, "__iter__"):
                        return torch.as_tensor(infer.audio)
                    # KPipeline packs text into <=context_length phoneme
                    # buckets and yields one result per bucket. Consume ALL
                    # of them; taking only the first silently drops audio.
                    parts = [torch.as_tensor(r.audio) for r in infer]
                    if not parts:
                        return None
                    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

            # Run blocking inference off the main loop
            audio = await asyncio.to_thread(infer_on_chunk)
            if audio is None:
                continue

            yield StreamChunk(
                audio=audio,
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
