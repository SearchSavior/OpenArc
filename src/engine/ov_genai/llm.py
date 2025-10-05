import asyncio
import gc
import logging
from typing import Any, AsyncIterator, Dict, List, Union

import openvino as ov
from openvino_genai import (
    GenerationConfig,
    LLMPipeline,
)
from transformers import AutoTokenizer

from src.server.models.ov_genai import OVGenAI_GenConfig
from src.server.model_registry import ModelLoadConfig, ModelRegistry
from src.engine.ov_genai.streamers import ChunkStreamer

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OVGenAI_LLM:
    def __init__(self, load_config: ModelLoadConfig):
        self.model_path = None
        self.encoder_tokenizer = None
        self.load_config = load_config

    def prepare_inputs(self, 
        messages: List[Dict[str, str]], 
        tools: List[Dict[str, Any]] = []) -> ov.Tensor:
        """
        Convert a messages (list of {role, content}) into ov.Tensor using the cached AutoTokenizer
        and its chat template.

        apply_chat_template can be configured to return a numpy array, 
        which we then convert to an ov.Tensor the runtime can accept

        Args:
            messages: List[Dict[str, str]]
            tools: List[Dict[str, Any]] - List of tools/functions available to the model

        returns:
            prompt_token_ids: 
        """
        prompt_token_ids = self.encoder_tokenizer.apply_chat_template(
            messages, 
            tools=tools if tools else None,
            add_generation_prompt=True,
            skip_special_tokens=True,
            return_tensors="np"
            )
        return ov.Tensor(prompt_token_ids)
    
    def generate_type(self, gen_config: OVGenAI_GenConfig):
        """
        Unified text generation method that routes to streaming or non-streaming
        based on the stream flag in gen_config. Both paths return an async iterator.
        
        Args:
            gen_config: Configuration containing the stream flag and other parameters
            
        Returns:
            - Non-streaming: async iterator yielding [metrics: dict, new_text: str]
            - Streaming: async iterator yielding token chunks (str)... then [metrics: dict, new_text: str]
        """
        if gen_config.stream:
            return self.generate_stream(gen_config)
        else:
            return self.generate_text(gen_config)
    
    async def generate_text(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """
        Async non-streaming text generation.
        Yields in order: metrics (dict), new_text (str).
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        prompt_token_ids = self.prepare_inputs(gen_config.messages, gen_config.tools)
        result = await asyncio.to_thread(self.model.generate, prompt_token_ids, generation_kwargs)
        
        perf_metrics = result.perf_metrics
        decoder_tokenizer = self.model.get_tokenizer()
        text = decoder_tokenizer.decode(result.tokens)[0] if getattr(result, "tokens", None) else ""

        metrics_dict = self.collect_metrics(gen_config, perf_metrics)
        yield metrics_dict
        yield text

    async def generate_stream(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Async streaming text generation.
        Yields token chunks (str) as they arrive, then metrics (dict), then final new_text (str).
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty
        )

        decoder_tokenizer = self.model.get_tokenizer()
        streamer = ChunkStreamer(decoder_tokenizer, gen_config)
        prompt_token_ids = self.prepare_inputs(gen_config.messages, gen_config.tools)

        async def _run_generation():
            return await asyncio.to_thread(
                self.model.generate,
                prompt_token_ids,
                generation_kwargs,
                streamer
            )

        gen_task = asyncio.create_task(_run_generation())

        try:
            while True:
                chunk = await streamer.text_queue.get()
                if chunk is None:
                    break
                yield chunk

        finally:
            result = await gen_task
            perf_metrics = result.perf_metrics
            metrics = self.collect_metrics(gen_config, perf_metrics)
            
            yield metrics
    
    def collect_metrics(self, gen_config: OVGenAI_GenConfig, perf_metrics) -> Dict[str, Any]:
        """
        Collect and format performance metrics into a dictionary.
        
        Args:
            gen_config: OVGenAI_GenConfig
            perf_metrics: PerfMetrics

        Returns:
            metrics: Dict[str, Any]
            """
        # Compute prefill throughput = input tokens / ttft (in seconds)
        # Inspired by section 2.2 (https://arxiv.org/pdf/2404.14294v3)
        ttft_seconds = perf_metrics.get_ttft().mean / 1000
        input_tokens = perf_metrics.get_num_input_tokens()
        prefill_throughput = round(input_tokens / ttft_seconds, 2) if ttft_seconds > 0 else 0

        metrics: Dict[str, Any] = {
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 5),
            'prefill_throughput (tokens/s)': prefill_throughput,
            'decode_throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 5),
            'decode_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 5),
            'input_token': input_tokens,
            'new_token': perf_metrics.get_num_generated_tokens(),
            'total_token': input_tokens + perf_metrics.get_num_generated_tokens(),
            'stream': gen_config.stream,
        }
        # Include streaming-specific fields
        if gen_config.stream and hasattr(gen_config, "stream_chunk_tokens"):
            metrics['stream_chunk_tokens'] = gen_config.stream_chunk_tokens
        
        return metrics

    def load_model(self, loader: ModelLoadConfig):
        """Load model using a ModelLoadConfig configuration and cache the tokenizer.

        Args:
            loader: ModelLoadConfig containing model_path, device, engine, and runtime_config.
        """

        self.model = LLMPipeline(
            loader.model_path,
            loader.device,
            **(loader.runtime_config or {})
        )

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(loader.model_path)
        logging.info(f"Model loaded successfully: {loader.model_name}")

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister model from registry and free memory resources.

        Args:
            registry: ModelRegistry to unregister from
            model_id: Private model identifier returned by register_load

        Returns:
            True if the model was found and unregistered, else False.
        """
        removed = await registry.register_unload(model_name)

        if self.model is not None:
            del self.model
            self.model = None
        
        if self.encoder_tokenizer is not None:
            del self.encoder_tokenizer
            self.encoder_tokenizer = None
        
        gc.collect()
        logging.info(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed


