import asyncio
import gc
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import openvino as ov
from openvino_genai import (
    GenerationConfig,
    LLMPipeline,
)
from transformers import AutoTokenizer

from src.server.models.ov_genai import OVGenAI_GenConfig
from src.server.model_registry import ModelRegistry
from src.server.models.registration import ModelLoadConfig
from src.engine.ov_genai.streamers import ChunkStreamer, BlockStreamer
from src.server.utils.chat import flatten_messages

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
        # Track active streaming requests for cancellation
        self._active_streamer: Optional[ChunkStreamer] = None
        self._active_request_id: Optional[str] = None

    def prepare_inputs(self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None) -> ov.Tensor:
        """
        Convert a messages (list of {role, content}) into ov.Tensor using the cached AutoTokenizer
        and its chat template.

        apply_chat_template can be configured to return a numpy array, 
        which we then convert to an ov.Tensor the runtime can accept

        Args:
            messages: List[Dict[str, Any]]
            tools: Optional[List[Dict[str, Any]]] - List of tools/functions available to the model

        returns:
            prompt_token_ids: 
        """
        prompt_token_ids = self.encoder_tokenizer.apply_chat_template(
            flatten_messages(messages),
            tools=tools,
            add_generation_prompt=True,
            skip_special_tokens=True,
            return_tensors="np"
            )
        return ov.Tensor(prompt_token_ids)

    async def cancel(self, request_id: str) -> bool:
        """
        Cancel an ongoing streaming generation by request_id.

        Args:
            request_id: The request ID to cancel

        Returns:
            True if cancellation was triggered, False if request_id didn't match
        """
        if self._active_request_id == request_id and self._active_streamer is not None:
            self._active_streamer.cancel()
            logger.info(f"[{self.load_config.model_name}] Cancellation triggered for request {request_id}")
            return True
        return False

    async def arc_infer(self, gen_config: OVGenAI_GenConfig, request_id: Optional[str] = None) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Unified inference method that uses appropriate streamer based on stream flag.
        - stream=True: Uses ChunkStreamer for incremental token streaming
        - stream=False: Uses BlockStreamer for single-block output

        Args:
            gen_config: Configuration containing generation parameters including stream flag
            request_id: Optional request ID for tracking cancellation

        Yields:
            Token chunks (str) as they arrive, then metrics (dict) at the end.
            For non-streaming (stream=False), yields a single chunk with all tokens.
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        decoder_tokenizer = self.model.get_tokenizer()

        # Select appropriate streamer based on stream flag
        if gen_config.stream:
            # Streaming mode: use ChunkStreamer with configured chunk size
            from copy import deepcopy
            streamer_config = deepcopy(gen_config)
            streamer_config.stream_chunk_tokens = gen_config.stream_chunk_tokens
            streamer = ChunkStreamer(decoder_tokenizer, streamer_config)
        else:
            # Non-streaming mode: use BlockStreamer for single-block output
            streamer = BlockStreamer(decoder_tokenizer)

        # Track active streamer for cancellation
        self._active_streamer = streamer
        self._active_request_id = request_id

        # Support pre-encoded input_ids, raw prompts, and chat messages
        if gen_config.input_ids:
            # Pre-encoded input IDs (used by /openarc/bench endpoint for benchmarking)
            import numpy as np
            prompt_token_ids = ov.Tensor(np.array(gen_config.input_ids, dtype=np.int64).reshape(1, -1))
        elif gen_config.prompt:
            # Direct tokenization for raw text (used by /v1/completions endpoint)
            prompt_token_ids = ov.Tensor(self.encoder_tokenizer.encode(gen_config.prompt, return_tensors="np"))
        else:
            # Chat template tokenization for messages (used by /v1/chat/completions endpoint)
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
            # Clear active streamer tracking
            self._active_streamer = None
            self._active_request_id = None
            # Wait for generation task to complete (may be cancelled)
            try:
                result = await gen_task
                perf_metrics = result.perf_metrics
                metrics = self.collect_metrics(gen_config, perf_metrics)
                yield metrics
            except Exception:
                # Generation was cancelled or failed, don't yield metrics
                pass

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
        
        logger.info(f"{loader.model_name} loading...")
        logger.info(f"{loader.model_type} on {loader.device} with {loader.runtime_config}")

        self.model = LLMPipeline(
            loader.model_path,
            loader.device,
            **(loader.runtime_config or {})
        )

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(loader.model_path)
        logging.info(f"{loader.model_name} loaded successfully")

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
        logging.info(f"[{self.load_config.model_name}] unloaded successfully")
        return removed
