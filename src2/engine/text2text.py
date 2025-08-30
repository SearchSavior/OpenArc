import gc
import asyncio
import json
import queue
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, field

from openvino_genai import (
    GenerationConfig, 
    LLMPipeline,
    TokenizedInputs
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
    def __init__(self, loader_config: OVGenAI_LoadConfig):
        self.id_model = None
        self.loader_config = loader_config
        self.current_generation: Optional[GenerationResult] = None

    def prepare_inputs(self, conversation: List[Dict[str, str]]) -> TokenizedInputs:
        """
        Convert a conversation (list of {role, content}) into TokenizedInputs using the tokenizer
        and its chat template.
        """
        tokenizer = self.id_model.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            conversation, 
            add_generation_prompt=True,
            
            )
        inputs: TokenizedInputs = tokenizer.encode(
            prompt, 
            add_special_tokens=True, 
            pad_to_max_length=False, 
            )
        return inputs
    
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
    
    async def _generate_text(self, gen_config: OVGenAI_TextGenConfig) -> GenerationResult:
        """
        Async non-streaming text generation.
        """
        ov_gen_cfg = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        inputs = self.prepare_inputs(gen_config.conversation)
        # Run the blocking id_model.generate call in a thread pool
        result = await asyncio.to_thread(self.id_model.generate, inputs, ov_gen_cfg)
        
        perf_metrics = result.perf_metrics
        # Decode first sequence
        tokenizer = self.id_model.get_tokenizer()
        text = tokenizer.decode(result.tokens)[0] if getattr(result, "tokens", None) else ""

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
        result_obj = GenerationResult(text=text, metrics=metrics_dict)
        self.current_generation = result_obj
        return result_obj

    async def _generate_stream(self, gen_config: OVGenAI_TextGenConfig) -> AsyncIterator[str]:
        """
        Async streaming text generation that yields text chunks suitable for FastAPI streaming.
        After streaming completes, `self.current_generation` contains final text and metrics.
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty
        )

        tokenizer = self.id_model.get_tokenizer()
        streamer = ChunkStreamer(tokenizer, gen_config)
        inputs = self.prepare_inputs(gen_config.conversation)
        self.current_generation = GenerationResult()

        async def _run_generation():
            return await asyncio.to_thread(
                self.id_model.generate,
                inputs,
                generation_kwargs,
                streamer
            )

        gen_task = asyncio.create_task(_run_generation())

        try:
            while True:
                chunk = await asyncio.to_thread(streamer.text_queue.get)
                if chunk is None:
                    break
                if self.current_generation is not None:
                    self.current_generation.chunks.append(chunk)
                yield chunk
        finally:
            result = await gen_task
            perf_metrics = result.perf_metrics
            metrics_dict = {
                'stream': gen_config.stream,
                'stream_chunk_tokens': gen_config.stream_chunk_tokens,
                'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
                'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
                'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
                'throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
                'generate_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2),
                'input_tokens': perf_metrics.get_num_input_tokens(),
                'new_tokens': perf_metrics.get_num_generated_tokens(),
                'total_tokens': perf_metrics.get_num_input_tokens() + perf_metrics.get_num_generated_tokens(),
            }
            # Decode final sequence into text and store metrics
            if self.current_generation is not None:
                self.current_generation.text = (tokenizer.decode(result.tokens)[0] if getattr(result, "tokens", None) else None)
                self.current_generation.metrics = metrics_dict
    
    def load_model(self):
        """
        Loads an OpenVINO GenAI text-to-text model.
        """
        self.id_model = LLMPipeline(
            self.loader_config.id_model,
            self.loader_config.device,
            **(self.loader_config.properties or {})
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
            id_model="/mnt/Ironwolf-4TB/Models/OpenVINO/Phi/phi-4-int4_asym-awq-ov",
            device="GPU.2"
        )

        conversation_messages = [
            {"role": "system", "content": "Alway's talk like you are Pete, the succint, punctual and self-deprecating pirate captain."},
            {"role": "user", "content": "Man it stinks in here"},
            {"role": "assistant", "content": "Arrr matey! The stench be foul, but we'll be smelling the sea air soon enough."},
            {"role": "user", "content": "You bet. Hey, thanks for the lift. What'd you say your name was?"}
        ]

        textgeneration_gen_config = OVGenAI_TextGenConfig(
            conversation=conversation_messages,
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
        if text_gen.current_generation:
            print(json.dumps(text_gen.current_generation.metrics or {}, indent=2))


    asyncio.run(_demo())
