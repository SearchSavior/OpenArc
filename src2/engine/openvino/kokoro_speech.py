# implementation adapted from 
# https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/kokoro/kokoro.ipynb




import json
from pathlib import Path
import soundfile as sf
import asyncio
from typing import AsyncIterator, NamedTuple

import torch
import openvino as ov
from kokoro.model import KModel
from kokoro.pipeline import KPipeline   
from pydantic import BaseModel, Field


class StreamChunk(NamedTuple):
    audio: torch.FloatTensor
    chunk_text: str
    chunk_index: int
    total_chunks: int


class OV_KokoroLoadConfig(BaseModel):
    model_dir: Path = Field(..., description="Model directory")
    device: str = Field(..., description="Device to use for generation")

class OV_KokoroGenConfig(BaseModel):
    input_text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(..., description="Voice to use for generation")
    speed: float = Field(1.0, description="Speed of the speech")
    character_count_chunk: int = Field(300, description="Number of characters per chunk")



class OVKModel(KModel):
    def __init__(self, config: OV_KokoroLoadConfig):
        super().__init__()

        self.model_dir = config.model_dir
        self._device = config.device  # Use _device instead of device

        # Load config.json
        with (self.model_dir / "config.json").open("r", encoding="utf-8") as f:
            model_config = json.load(f)

        self.vocab = model_config["vocab"]
        self.context_length = model_config["plbert"]["max_position_embeddings"]

        # Compile OpenVINO model
        core = ov.Core()
        self.model = core.compile_model(self.model_dir / "openvino_model.xml", self._device)


    def generate_audio(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        config: OV_KokoroGenConfig
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        outputs = self.model([input_ids, ref_s, torch.tensor(config.speed)])
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])

    def kokoro_forward(self, pipeline: KPipeline, config: OV_KokoroGenConfig):
        """
        Encapsulates torch.no_grad and pipeline call for text-to-speech generation.
        
        Args:
            pipeline: KPipeline instance
            config: Generation configuration containing input_text and voice
            
        Returns:
            Generated audio result
        """
        with torch.no_grad():
            generator = pipeline(config.input_text, voice=config.voice)
            result = next(generator)
        return result

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks based on character count, trying to break at sentence boundaries.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back
            sentence += '.'
            
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Sentence is longer than chunk_size, split by words
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = word
                            else:
                                # Single word longer than chunk_size, force split
                                chunks.append(word[:chunk_size])
                                temp_chunk = word[chunk_size:]
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    current_chunk = temp_chunk
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    async def kokoro_stream(self, pipeline: KPipeline, config: OV_KokoroGenConfig) -> AsyncIterator[StreamChunk]:
        """
        Stream audio generation by chunking input text based on character count.
        
        Args:
            pipeline: KPipeline instance
            config: Generation configuration containing input_text, voice, and chunk size
            
        Yields:
            StreamChunk containing audio data, chunk text, and metadata
        """
        # Chunk the input text
        text_chunks = self._chunk_text(config.input_text, config.character_count_chunk)
        total_chunks = len(text_chunks)
        
        for chunk_index, chunk_text in enumerate(text_chunks):
            # Create config for this chunk
            chunk_config = OV_KokoroGenConfig(
                input_text=chunk_text,
                voice=config.voice,
                speed=config.speed,
                character_count_chunk=config.character_count_chunk
            )
            
            # Generate audio for this chunk
            with torch.no_grad():
                generator = pipeline(chunk_text, voice=config.voice)
                result = next(generator)
            
            # Yield the chunk result
            yield StreamChunk(
                audio=result.audio,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                total_chunks=total_chunks
            )
            
            # Allow other async tasks to run
            await asyncio.sleep(0)


# =====================================================================
# Example usage
# =====================================================================
if __name__ == "__main__":
    # Create load configuration
    load_config = OV_KokoroLoadConfig(
        model_dir=Path("/mnt/Ironwolf-4TB/Models/OpenVINO/Kokoro-82M-FP16-OpenVINO"),
        device="CPU"
    )

    # Initialize model + pipeline
    ov_model = OVKModel(load_config)
    pipeline = KPipeline(model=ov_model, lang_code="a")

    # Create generation configuration
    with open("/home/echo/Projects/OpenArc/src2/tests/test_kokoro.txt", "r") as f:
        test_text = f.read()

    gen_config = OV_KokoroGenConfig(
        input_text=test_text,
        voice="af_heart",
        speed=1.0
    )

    # Example 1: Single forward pass
    result = ov_model.kokoro_forward(pipeline, gen_config)
    print(f"Generated audio with {len(result.audio)} samples at 24kHz")
    
    # Save as WAV file
    output_path = "kokoro_output.wav"
    sf.write(output_path, result.audio, 24000)  # 24kHz sample rate
    print(f"Audio saved to: {output_path}")
    
    # Example 2: Streaming generation
    async def stream_example():
        print("\n--- Streaming Example ---")
        all_audio = []
        
        async for chunk in ov_model.kokoro_stream(pipeline, gen_config):
            print(f"Generated chunk {chunk.chunk_index + 1}/{chunk.total_chunks}: "
                  f"{len(chunk.chunk_text)} chars, {len(chunk.audio)} samples")
            print(f"Chunk text: {chunk.chunk_text[:50]}...")
            all_audio.append(chunk.audio)
        
        # Concatenate all audio chunks
        if all_audio:
            import numpy as np
            combined_audio = np.concatenate(all_audio)
            stream_output_path = "kokoro_stream_output.wav"
            sf.write(stream_output_path, combined_audio, 24000)
            print(f"Streaming audio saved to: {stream_output_path}")
    
    # Run the streaming example
    asyncio.run(stream_example())
