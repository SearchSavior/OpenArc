# streaming_kokoro_async.py
"""
Streaming-only Kokoro + OpenVINO implementation.
Now uses asyncio.to_thread for non-blocking streaming inference.
"""

import asyncio
import gc
import json
import re
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, NamedTuple

import openvino as ov
import soundfile as sf
import torch
from kokoro.model import KModel


from src.server.model_registry import ModelLoadConfig, ModelRegistry
from src.server.models.openvino import KokoroLanguage, KokoroVoice, OV_KokoroGenConfig


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
            model_id: Private model identifier returned by register_load

        Returns:
            True if the model was found and unregistered, else False.
        """
        removed = await registry.register_unload(model_name)

        gc.collect()
        return True


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

        text_chunks = self.make_chunks(config.kokoro_message, config.character_count_chunk)
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

async def demo_entrypoint():
    """
    Demo entrypoint: Load OV_Kokoro model, generate speech, save to WAV, and unload.
    """
    import sys

    # Add the project root to Python path for imports
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src2.server.model_registry import EngineType, ModelLoadConfig, TaskType

    # Example configuration - adjust paths and parameters as needed
    model_path = Path("/mnt/Ironwolf-4TB/Models/OpenVINO/Kokoro-82M-FP16-OpenVINO")  # Replace with actual model path
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        print("Please update the model_path in demo_entrypoint()")
        return

    load_config = ModelLoadConfig(
        model_path=str(model_path),
        model_name="kokoro-demo",
        model_type=TaskType.KOKORO,
        engine=EngineType.OPENVINO,
        device="CPU",  # or "GPU" if available
    )

    # Create model instance
    kokoro_model = OV_Kokoro(load_config)

    try:
        # Load the model
        print("Loading Kokoro model...")
        kokoro_model.load_model(load_config)
        print("Model loaded successfully")

        # Configure generation
        gen_config = OV_KokoroGenConfig(
            kokoro_message="Hello world! This is a test of Kokoro text-to-speech synthesis.",
            voice=KokoroVoice.AF_SARAH,  # American English female voice
            lang_code=KokoroLanguage.AMERICAN_ENGLISH,
            speed=1.0,
            character_count_chunk=100,
            response_format="wav"
        )

        # Generate speech
        print("Generating speech...")
        audio_chunks = []
        async for chunk in kokoro_model.chunk_forward_pass(gen_config):
            print(f"Generated chunk {chunk.chunk_index + 1}/{chunk.total_chunks}: '{chunk.chunk_text}'")
            audio_chunks.append(chunk.audio)

        # Concatenate all audio chunks
        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=0)
            print(f"Generated audio with shape: {full_audio.shape}")

            # Save to WAV file
            output_path = Path("kokoro_output.wav")
            sf.write(str(output_path), full_audio.numpy(), samplerate=24000)  # Kokoro uses 24kHz
            print(f"Audio saved to {output_path}")

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Unload the model
        print("Unloading model...")
        await kokoro_model.unload_model()
        print("Model unloaded")


if __name__ == "__main__":
    asyncio.run(demo_entrypoint())



