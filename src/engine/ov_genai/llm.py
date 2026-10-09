from src.engine.ov_genai.utils import (
    apply_temperature,
    check_vram_budget,
    extract_scheduler_config_from_loader,
    format_perf_metrics,
    load_draft_model,
)
import asyncio
import gc
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import os

import openvino as ov
from openvino_genai import (
    GenerationConfig,
    LLMPipeline,
)
from transformers import AutoTokenizer, BatchEncoding

from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import ModelLoadConfig
from src.engine.ov_genai.streamers import ensure_tool_call_parser, select_streamer
from src.server.utils.chat import flatten_messages

logger = logging.getLogger(__name__)


class OVGenAI_LLM:
    def __init__(self, load_config: ModelLoadConfig):
        self.model_path = None
        self.encoder_tokenizer = None
        self.load_config = load_config
        self._active_request_id: Optional[str] = None
        self._active_streamer: Optional[ChunkStreamer] = None

    def prepare_inputs(self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        chat_template_kwargs: dict = {}
    ) -> ov.Tensor:
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
            return_tensors="np",
            **chat_template_kwargs,
            )
        if isinstance(prompt_token_ids, BatchEncoding):
            prompt_token_ids = prompt_token_ids['input_ids']
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
        ensure_tool_call_parser(gen_config, self.load_config)
        generation_kwargs = self.create_generation_config(gen_config)

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
            prompt_token_ids = self.prepare_inputs(gen_config.messages, gen_config.tools, gen_config.chat_template_kwargs)
        
        result = await asyncio.to_thread(self.model.generate, prompt_token_ids, generation_kwargs)
        
        perf_metrics = result.perf_metrics
        decoder_tokenizer = self.model.get_tokenizer()
        # gemma4/museglimmer protocol tags are special=True: keep them in the
        # decoded text so the route-level parse_generation can split
        # reasoning/tool calls.
        keep_special = getattr(gen_config, "tool_call_parser", None) in (
            "gemma4", "museglimmer",
        )
        text = (
            decoder_tokenizer.decode(result.tokens, skip_special_tokens=not keep_special)[0]
            if getattr(result, "tokens", None) else ""
        )

        metrics_dict = self.collect_metrics(gen_config, perf_metrics)
        yield metrics_dict
        yield text

    async def generate_stream(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Async streaming text generation.
        Yields token chunks (str) as they arrive, then metrics (dict), then final new_text (str).
        """
        ensure_tool_call_parser(gen_config, self.load_config)
        generation_kwargs = self.create_generation_config(gen_config)
        decoder_tokenizer = self.model.get_tokenizer()
        streamer = select_streamer(decoder_tokenizer, gen_config)
        
        # Track active request and streamer for cancellation
        self._active_request_id = gen_config.request_id
        self._active_streamer = streamer
        
        # Support both chat messages and raw prompts
        if gen_config.prompt:
            # Direct tokenization for raw text (used by /v1/completions endpoint)
            prompt_token_ids = ov.Tensor(self.encoder_tokenizer.encode(gen_config.prompt, return_tensors="np"))
        else:
            # Chat template tokenization for messages (used by /v1/chat/completions endpoint)
            prompt_token_ids = self.prepare_inputs(gen_config.messages, gen_config.tools, gen_config.chat_template_kwargs)

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
            # Clear active request tracking
            self._active_request_id = None
            self._active_streamer = None
            
            result = await gen_task
            perf_metrics = result.perf_metrics
            metrics = self.collect_metrics(gen_config, perf_metrics)
            
            yield metrics
    
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
    
    def collect_metrics(self, gen_config: OVGenAI_GenConfig, perf_metrics) -> Dict[str, Any]:
        """
        Collect and format performance metrics into a dictionary.

        Args:
            gen_config: OVGenAI_GenConfig
            perf_metrics: PerfMetrics

        Returns:
            metrics: Dict[str, Any]
            """
        return format_perf_metrics(gen_config, perf_metrics)

    def load_model(self, loader: ModelLoadConfig):
        """Load model using a ModelLoadConfig configuration and cache the tokenizer.

        Args:
            loader: ModelLoadConfig containing model_path, device, engine, and runtime_config.
        """
        
        logger.info(f"{loader.model_name} loading...")
        logger.info(f"{loader.model_type} on {loader.device} with {loader.runtime_config}")

        check_vram_budget(loader)

        # Load draft model for speculative decoding if provided
        (
            draft_model,
            self.draft_model_loaded,
            self.model_num_assistant_tokens,
            self.model_assistant_confidence_threshold,
        ) = load_draft_model(loader)
        
        pipeline_kwargs = {**(loader.runtime_config or {})}
        scheduler_config = extract_scheduler_config_from_loader(loader)
        if loader.cache_dir:
            pipeline_kwargs['CACHE_DIR'] = loader.cache_dir
        if draft_model is not None:
            pipeline_kwargs['draft_model'] = draft_model
        
        self.model = LLMPipeline(
            loader.model_path,
            loader.device,
            **scheduler_config,
            **pipeline_kwargs,
        )

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(loader.model_path)
        logging.info(f"{loader.model_name} loaded successfully")

        # Warm up: one short generation so OpenVINO compiles its GPU kernels
        # at load time instead of on the first request. OPENARC_WARMUP=0 opts out.
        if os.getenv('OPENARC_WARMUP', '1') != '0':
            try:
                warm_cfg = self.model.get_generation_config()
                warm_cfg.max_new_tokens = 8
                warm_cfg.do_sample = False
                self.model.generate("Hello", warm_cfg)
                logger.info(f"{loader.model_name} warm-up generation complete")
            except Exception as e:
                logger.warning(f"{loader.model_name} warm-up generation failed: {e}")

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

    def create_generation_config(self, config: OVGenAI_GenConfig) -> GenerationConfig:
        """
        Converts the config received by the API to the OpenVino-compatible config.
        """
        generation_kwargs = self.model.get_generation_config() if self.model else GenerationConfig()
        generation_kwargs.max_new_tokens = config.max_tokens
        apply_temperature(generation_kwargs, config.temperature)
        generation_kwargs.top_k = config.top_k
        generation_kwargs.top_p = config.top_p
        generation_kwargs.repetition_penalty = config.repetition_penalty

        if config.seed:
            generation_kwargs.rng_seed = config.seed
        if config.frequency_penalty:
            generation_kwargs.frequency_penalty = config.frequency_penalty
        if config.presence_penalty:
            generation_kwargs.presence_penalty = config.presence_penalty
            
        # Add speculative decoding parameters (mutually exclusive per OpenVINO docs)
        if config.num_assistant_tokens is not None:
            generation_kwargs.num_assistant_tokens = config.num_assistant_tokens
        elif config.assistant_confidence_threshold is not None:
            generation_kwargs.assistant_confidence_threshold = config.assistant_confidence_threshold
        elif getattr(self, 'draft_model_loaded', False):
            if self.model_num_assistant_tokens is not None:
                generation_kwargs.num_assistant_tokens = self.model_num_assistant_tokens
            elif self.model_assistant_confidence_threshold is not None:
                generation_kwargs.assistant_confidence_threshold = self.model_assistant_confidence_threshold
            else:
                default_tokens = int(os.getenv('OPENARC_DEFAULT_NUM_ASSISTANT_TOKENS', '4'))
                generation_kwargs.num_assistant_tokens = default_tokens
        return generation_kwargs
