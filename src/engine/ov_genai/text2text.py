import gc
import asyncio
import json
import queue
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Union, AsyncIterator

from openvino_genai import (
    GenerationConfig, 
    LLMPipeline
    )

from engine.ov_genai.base_config import (
    OVGenAI_LoadConfig, 
    OVGenAI_TextGenConfig
    )
from engine.ov_genai.streamers import ChunkStreamer


class OVGenAI_Text2Text:
    def __init__(self, loader_config: OVGenAI_LoadConfig):
        self.model = None
        self.loader_config = loader_config
        self.last_metrics: Optional[Dict[str, Any]] = None
        self.last_text: Optional[str] = None

    def load_model(self):
        """
        Loads an OpenVINO GenAI text-to-text model.
        """
        self.model = LLMPipeline(
            self.loader_config.id_model,
            self.loader_config.device,
            **(self.loader_config.properties or {})
        )
        print("Model loaded successfully.")
    
    def generate_response(self, gen_config: OVGenAI_TextGenConfig):
        """
        Unified text generation method that routes to streaming or non-streaming 
        based on the stream flag in gen_config.
        
        Args:
            gen_config: Configuration containing the stream flag and other parameters
            
        Returns:
            - Non-streaming: Tuple of (metrics_dict, generated_text)
            - Streaming: Async iterator of string chunks
        """
        if gen_config.stream:
            return self.generate_stream(gen_config)
        else:
            return self.generate_text(gen_config)
    
    def generate_text(self, gen_config: OVGenAI_TextGenConfig) -> str:
        """
        Non-streaming text generation.
        """
        generation_gen_config = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        result = self.model.generate([gen_config.conversation], generation_gen_config)
        perf_metrics = result.perf_metrics

        metrics_dict = {
            'stream' : gen_config.stream,
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
            'generate_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2),
            'input_tokens': perf_metrics.get_num_input_tokens(),
            'new_tokens': perf_metrics.get_num_generated_tokens(),
            'total_tokens': perf_metrics.get_num_input_tokens() + perf_metrics.get_num_generated_tokens(),
        }
        return metrics_dict, result.texts[0]

    async def generate_stream(self, gen_config: OVGenAI_TextGenConfig) -> AsyncIterator[str]:
        """
        Async streaming text generation that yields text chunks suitable for FastAPI streaming.
        After streaming completes, metrics are stored in `self.last_metrics` and final text in `self.last_text`.
        """
        generation_gen_config = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        tokenizer = self.model.get_tokenizer()
        streamer = ChunkStreamer(tokenizer, gen_config)

        async def _run_generation():
            return await asyncio.to_thread(
                self.model.generate,
                [gen_config.conversation],
                generation_gen_config,
                streamer
            )

        gen_task = asyncio.create_task(_run_generation())

        try:
            while True:
                chunk = await asyncio.to_thread(streamer.text_queue.get)
                if chunk is None:
                    break
                yield chunk
        finally:
            result = await gen_task
            perf_metrics = result.perf_metrics
            self.last_metrics = {
                'stream': gen_config.stream,
                'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
                'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
                'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
                'throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
                'generate_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2),
                'input_tokens': perf_metrics.get_num_input_tokens(),
                'new_tokens': perf_metrics.get_num_generated_tokens(),
                'total_tokens': perf_metrics.get_num_input_tokens() + perf_metrics.get_num_generated_tokens(),
            }
            # Prefer engine-produced final text if present; fallback to concatenation already emitted.
            self.last_text = result.texts[0] if result.texts else None

    def unload_model(self):
        """Unload model and free memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        gc.collect()
        print("Model unloaded and memory cleaned up")
# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    async def _demo():
        load_cfg = OVGenAI_LoadConfig(
            id_model="/mnt/Ironwolf-4TB/Models/OpenVINO/Phi/Phi-lthy4-OpenVINO/Phi-lthy4-int4_sym-awq-ov",
            device="GPU.1"
        )

        conversation_messages = [
            {"role": "system", "content": "Alway's talk like you are Pete, the succint, punctual and self-deprecating pirate captain."},
            {"role": "user", "content": "Man it stinks in here"},
            {"role": "assistant", "content": "Arrr matey! The stench be foul, but we'll be smelling the sea air soon enough."},
            {"role": "user", "content": "You bet. Hey, thanks for the lift. What'd you say your name was?"}
        ]

        conversation_json = json.dumps(conversation_messages, indent=2)

        textgeneration_gen_config = OVGenAI_TextGenConfig(
            conversation=conversation_json,
            max_new_tokens=128,
            temperature=1.2,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            stream_chunk_tokens=1,
            stream=True
        )

        text_gen = OVGenAI_Text2Text(load_cfg)
        text_gen.load_model()

        async for chunk in text_gen.generate_stream(textgeneration_gen_config):
            print(chunk, end="", flush=True)

        print("\n\nPerformance Metrics")
        print("-"*20)
        print(json.dumps(text_gen.last_metrics or {}, indent=2))

    asyncio.run(_demo())
