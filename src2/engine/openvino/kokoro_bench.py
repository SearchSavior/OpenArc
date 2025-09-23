# streaming_kokoro_async.py
"""
Streaming-only Kokoro + OpenVINO implementation.
Now uses asyncio.to_thread for non-blocking streaming inference.
"""

import json
import asyncio
import re
from pathlib import Path
from typing import AsyncIterator, NamedTuple
from time import perf_counter

import numpy as np
import torch
import openvino as ov
import soundfile as sf
from pydantic import BaseModel, Field
from kokoro.model import KModel
from kokoro.pipeline import KPipeline



class StreamChunk(NamedTuple):
    audio: torch.Tensor
    chunk_text: str
    chunk_index: int
    total_chunks: int


# =====================================================================
# Config
# =====================================================================
class OV_KokoroLoadConfig(BaseModel):
    model_dir: Path = Field(..., description="Model directory containing config.json + IR")
    device: str = Field(..., description="OpenVINO device string (e.g., 'CPU', 'GPU')")


class OV_KokoroGenConfig(BaseModel):
    input_text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(..., description="Voice token")
    speed: float = Field(1.0, description="Speech speed multiplier")
    character_count_chunk: int = Field(100, description="Max characters per chunk")


# =====================================================================
# Model wrapper (streaming only, async inference via to_thread)
# =====================================================================
class OVKModel(KModel):
    def __init__(self, config: OV_KokoroLoadConfig):
        super().__init__()
        self.model_dir = config.model_dir
        self._device = config.device

        # Load model config.json (used by KPipeline)
        with (self.model_dir / "config.json").open("r", encoding="utf-8") as f:
            model_config = json.load(f)

        self.vocab = model_config["vocab"]
        self.context_length = model_config["plbert"]["max_position_embeddings"]

        # Compile OpenVINO IR once -> kept in memory
        core = ov.Core()
        self.model = core.compile_model(self.model_dir / "openvino_model.xml", self._device)

    # -----------------------------------------------------------------
    # Text chunking
    # -----------------------------------------------------------------
    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks up to `chunk_size` characters,
        preferring sentence boundaries.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # Regex: split after ., !, ? followed by space
        t_regex_start = perf_counter()
        sentences = re.split(r'(?<=[.!?]) +', text)
        t_regex_end = perf_counter()

        t_build_start = perf_counter()
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

        t_build_end = perf_counter()

        regex_ms = (t_regex_end - t_regex_start) * 1000.0
        build_ms = (t_build_end - t_build_start) * 1000.0
        total_ms = (t_build_end - t_regex_start) * 1000.0
        print(f"[Chunking] regex_split_ms={regex_ms:.3f} build_ms={build_ms:.3f} total_ms={total_ms:.3f} num_chunks={len(chunks)}")

        return chunks

    # -----------------------------------------------------------------
    # Streaming generator
    # -----------------------------------------------------------------
    async def kokoro_stream(
        self, pipeline: KPipeline, config: OV_KokoroGenConfig
    ) -> AsyncIterator[StreamChunk]:
        """
        Async generator yielding audio chunks from text.
        Uses asyncio.to_thread to offload inference calls.
        """
        t_stream_start = perf_counter()
        text_chunks = self._chunk_text(config.input_text, config.character_count_chunk)
        total_chunks = len(text_chunks)

        for idx, chunk_text in enumerate(text_chunks):

            def run_inference():
                """Blocking inference run in background thread."""
                with torch.no_grad():
                    t_inf_start = perf_counter()
                    gen = pipeline(chunk_text, voice=config.voice, speed=config.speed)
                    result = next(gen) if hasattr(gen, "__iter__") else gen
                    t_inf_end = perf_counter()
                    return result, (t_inf_end - t_inf_start)

            # Run blocking inference off the main loop
            result, infer_secs = await asyncio.to_thread(run_inference)
            print(f"[Inference] chunk={idx+1}/{total_chunks} latency_ms={infer_secs*1000.0:.3f} text_len={len(chunk_text)} samples={len(result.audio)}")

            yield StreamChunk(
                audio=result.audio,
                chunk_text=chunk_text,
                chunk_index=idx,
                total_chunks=total_chunks,
            )

        t_stream_end = perf_counter()
        print(f"[Generation] total_stream_time_ms={(t_stream_end - t_stream_start)*1000.0:.3f}")


# =====================================================================
# Example usage
# =====================================================================
if __name__ == "__main__":
    load_config = OV_KokoroLoadConfig(
        model_dir=Path("/mnt/Ironwolf-4TB/Models/OpenVINO/Kokoro-82M-FP16-OpenVINO"),
        device="CPU",
    )
    ov_model = OVKModel(load_config)
    pipeline = KPipeline(model=ov_model, lang_code="a")

    with open("/home/echo/Projects/OpenArc/src2/tests/test_kokoro.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        test_text = data.get("input_text", "")

    gen_config = OV_KokoroGenConfig(
        input_text=test_text,
        voice="af_heart",
        speed=1.0,
    )

    async def stream_example():
        print("\n--- Streaming Example ---")
        audio_chunks = []
        async for chunk in ov_model.kokoro_stream(pipeline, gen_config):
            print(f"[Chunk {chunk.chunk_index+1}/{chunk.total_chunks}] "
                  f"{len(chunk.chunk_text)} chars, {len(chunk.audio)} samples")
            audio_chunks.append(chunk.audio.cpu().numpy())

        if audio_chunks:
            t_concat_start = perf_counter()
            combined = np.concatenate(audio_chunks, axis=0)
            t_concat_end = perf_counter()
            print(f"[Post] concatenate_ms={(t_concat_end - t_concat_start)*1000.0:.3f}")
            sf.write("kokoro_stream_output.wav", combined, 24000)
            print("Saved kokoro_stream_output.wav")

    asyncio.run(stream_example())
