import gc
import asyncio
import json
import queue
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, field

from transformers import AutoTokenizer
import openvino as ov
from openvino_genai import (
    GenerationConfig, 
    LLMPipeline,
    )

from src2.api.base_config import (
    OVGenAI_LoadConfig, 
    OVGenAI_TextGenConfig
    )
from src2.engine.streamers import ChunkStreamer

@dataclass
class GenerationResult:
    """
    Results of a text generation.

    Args:
        text: Final decoded text. 
            - Used when stream is False.
        
        chunks: Collected stream chunks.
            - Used when stream is True.
        
        metrics: Performance metrics.
            
    """
    text: Optional[str] = None                       # Final decoded text
    chunks: List[str] = field(default_factory=list)  # Collected stream chunks
    metrics: Optional[Dict[str, Any]] = None         # Perf metrics

class OVGenAI_Text2Text:
    def __init__(self, load_config: OVGenAI_LoadConfig):
        self.id_model = None
        self.load_config = load_config
        self.generation_result: Optional[GenerationResult] = None

    def prepare_inputs(self, messages: List[Dict[str, str]]) -> ov.Tensor:
        """
        Convert a messages (list of {role, content}) into ov.Tensor using the AutoTokenizer
        and its chat template.
        """
        encoder_tokenizer = AutoTokenizer.from_pretrained(self.load_config.id_model)
        prompt_token_ids = encoder_tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            return_tensors="np"
            )
        return ov.Tensor(prompt_token_ids)
    
    def generate_type(self, gen_config: OVGenAI_TextGenConfig):
        """
        Unified text generation method that routes to streaming or non-streaming 
        based on the stream flag in gen_config.
        
        Args:
            gen_config: Configuration containing the stream flag and other parameters
            
        Returns:
            - Non-streaming: GenerationResult
            - Streaming: Async iterator of string chunks
        """
        if gen_config.stream:
            return self.generate_stream(gen_config)
        else:
            return self.generate_text(gen_config)
    
    def collect_metrics(self, gen_config: OVGenAI_TextGenConfig, perf_metrics) -> Dict[str, Any]:
        """
        Collect and format performance metrics into a dictionary.
        """
        # Compute prefill throughput = input tokens / ttft (in seconds)
        ttft_seconds = perf_metrics.get_ttft().mean / 1000
        input_tokens = perf_metrics.get_num_input_tokens()
        prefill_throughput = round((input_tokens / ttft_seconds), 2) if ttft_seconds > 0 else 0

        metrics: Dict[str, Any] = {
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
            'prefill_throughput (tokens/s)': prefill_throughput,
            'decode_throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
            'decode_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2),
            'input_token': input_tokens,
            'new_token': perf_metrics.get_num_generated_tokens(),
            'total_token': input_tokens + perf_metrics.get_num_generated_tokens(),
            'stream': gen_config.stream,
        }
        # Include streaming-specific fields
        if gen_config.stream and hasattr(gen_config, "stream_chunk_tokens"):
            metrics['stream_chunk_tokens'] = gen_config.stream_chunk_tokens
        
        return metrics
    
    async def generate_text(self, gen_config: OVGenAI_TextGenConfig) -> GenerationResult:
        """
        Async non-streaming text generation.
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        prompt_token_ids = self.prepare_inputs(gen_config.messages)
        # Run the blocking id_model.generate call in a thread pool
        result = await asyncio.to_thread(self.id_model.generate, prompt_token_ids, generation_kwargs)
        
        perf_metrics = result.perf_metrics
        # Decode first sequence
        decoder_tokenizer = self.id_model.get_tokenizer()
        text = decoder_tokenizer.decode(result.tokens)[0] if getattr(result, "tokens", None) else ""

        metrics_dict = self.collect_metrics(gen_config, perf_metrics)
        result_obj = GenerationResult(text=text, metrics=metrics_dict)
        self.generation_result = result_obj
        return result_obj

    async def generate_stream(self, gen_config: OVGenAI_TextGenConfig) -> AsyncIterator[str]:
        """
        
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty
        )

        decoder_tokenizer = self.id_model.get_tokenizer()
        streamer = ChunkStreamer(decoder_tokenizer, gen_config)
        prompt_token_ids = self.prepare_inputs(gen_config.messages)
        self.generation_result = GenerationResult()

        async def _run_generation():
            return await asyncio.to_thread(
                self.id_model.generate,
                prompt_token_ids,
                generation_kwargs,
                streamer
            )

        gen_task = asyncio.create_task(_run_generation())

        try:
            while True:
                chunk = await asyncio.to_thread(streamer.text_queue.get)
                if chunk is None:
                    break
                self.generation_result.chunks.append(chunk)
                yield chunk
        finally:
            result = await gen_task
            perf_metrics = result.perf_metrics
            metrics = self.collect_metrics(gen_config, perf_metrics)
            # Decode final sequence into text and store metrics
            self.generation_result.text = decoder_tokenizer.decode(result.tokens)[0]
            self.generation_result.metrics = metrics
    
    def load_model(self):
        """
        Loads an OpenVINO GenAI text-to-text model.
        """
        self.id_model = LLMPipeline(
            self.load_config.id_model,
            self.load_config.device,
            **(self.load_config.properties or {})
        )
        print("Model loaded successfully.")

    def unload_model(self):
        """Unload model and free memory"""
        if hasattr(self, 'id_model') and self.id_model is not None:
            del self.id_model
            self.id_model = None
        
        gc.collect()
        print("Model unloaded and memory cleaned up")
# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    async def _demo():
        load_cfg = OVGenAI_LoadConfig(
            id_model="/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Llama-3.2-3B-Instruct-abliterated-OpenVINO/Llama-3.2-3B-Instruct-abliterated-int4_asym-ov",
            device="GPU.2"
        )

        messages = [
            {"role": "system", "content": "Alway's talk like you are Pete, the succint, punctual and self-deprecating pirate captain."},
            {"role": "user", "content": "Man it stinks in here"},
            {"role": "assistant", "content": "Arrr matey! The stench be foul, but we'll be smelling the sea air soon enough."},
            {"role": "user", "content": "You bet. Hey, thanks for the lift. What'd you say your name was?"},
            {"role": "assistant", "content": "Arrr matey! The stench be foul, but we'll be smelling the sea air soon enough."},
            {"role": "user", "content": "You bet. Hey, thanks for the lift. What'd you say your name was?"}
        ]

        textgeneration_gen_config = OVGenAI_TextGenConfig(
            messages=messages,
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
        if text_gen.generation_result:
            print(json.dumps(text_gen.generation_result.metrics or {}, indent=2))


    asyncio.run(_demo())
