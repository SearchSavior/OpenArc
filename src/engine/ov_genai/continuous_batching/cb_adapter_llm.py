"""
TODO (required before this lands):
- Add `cb_config: ContinuousBatchSchedulerConfig` to ModelLoadConfig.
  This adapter reads scheduler settings via `loader.cb_config`.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional

import openvino as ov
from openvino_genai import (
    ContinuousBatchingPipeline,
    GenerationConfig,
    SchedulerConfig,
)
from transformers import AutoTokenizer, BatchEncoding

from src.engine.ov_genai.continuous_batching.cb_models import (
    ContinuousBatchSchedulerConfig,
)
from src.server.model_registry import ModelRegistry
from src.server.models.ov_genai import OVGenAI_GenConfig
from src.server.models.registration import ModelLoadConfig
from src.server.utils.chat import flatten_messages

logger = logging.getLogger(__name__)


class ArcCBLLM:
    """OpenArc adapter for OpenVINO GenAI continuous batching text models."""

    def __init__(self, load_config: ModelLoadConfig):
        self.load_config = load_config
        self.model: ContinuousBatchingPipeline | None = None
        self.encoder_tokenizer: AutoTokenizer | None = None

    def load_model(self, loader: ModelLoadConfig) -> None:
        """Load a ContinuousBatchingPipeline using the registry load contract."""

        logger.info("%s loading continuous batching pipeline...", loader.model_name)
        logger.info("%s on %s with %s", loader.model_type, loader.device, loader.runtime_config)

        # scheduler_config only works in a setattr loop we need to convert from pydantic
        scheduler = self._build_scheduler_config(loader.cb_config)

        self.model = ContinuousBatchingPipeline(
            loader.model_path,
            scheduler_config=scheduler,
            device=loader.device,
            properties=loader.runtime_config or {},
            tokenizer_properties={},
            vision_encoder_properties={},
        )
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(loader.model_path)

        logger.info("%s loaded successfully", loader.model_name)

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister the model and release pipeline resources."""

        removed = await registry.register_unload(model_name)
        self.model = None
        self.encoder_tokenizer = None
        gc.collect()
        logger.info("[%s] unloaded successfully", self.load_config.model_name)
        return removed

    def prepare_inputs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ov.Tensor:
        """Convert chat messages into an input_ids tensor using the cached AutoTokenizer."""
        prompt_token_ids = self.encoder_tokenizer.apply_chat_template(
            flatten_messages(messages),
            tools=tools,
            add_generation_prompt=True,
            skip_special_tokens=True,
            return_tensors="np",
        )
        # Recent transformers returns a BatchEncoding from apply_chat_template
        # when return_dict defaults to True; unwrap to the raw input_ids array.
        if isinstance(prompt_token_ids, BatchEncoding):
            prompt_token_ids = prompt_token_ids["input_ids"]
        return ov.Tensor(prompt_token_ids)

    def add_request(self, request_id: int, gen_config: OVGenAI_GenConfig):
        """Add one LLM request through the input_ids ContinuousBatchingPipeline overload."""
        request_input = self.prepare_inputs(gen_config.messages, gen_config.tools)
        generation_config = self.create_generation_config(gen_config)
        return self.model.add_request(request_id, request_input, generation_config)

    def create_generation_config(self, config: OVGenAI_GenConfig) -> GenerationConfig:
        generation_config = self.model.get_config()
        generation_config.max_new_tokens = config.max_tokens
        generation_config.temperature = config.temperature
        generation_config.top_k = config.top_k
        generation_config.top_p = config.top_p
        generation_config.repetition_penalty = config.repetition_penalty
        generation_config.rng_seed = config.seed
        generation_config.frequency_penalty = config.frequency_penalty
        generation_config.presence_penalty = config.presence_penalty
        return generation_config

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect ContinuousBatchingPipeline metrics."""
        metrics = self.model.get_metrics()
        return {
            "requests": metrics.requests,
            "scheduled_requests": metrics.scheduled_requests,
            "cache_usage": metrics.cache_usage,
            "max_cache_usage": metrics.max_cache_usage,
            "avg_cache_usage": metrics.avg_cache_usage,
            "kv_cache_size_in_bytes": metrics.kv_cache_size_in_bytes,
        }

    def _build_scheduler_config(
        self,
        config: ContinuousBatchSchedulerConfig,
    ) -> SchedulerConfig:
        scheduler = SchedulerConfig()
        for field, value in config.model_dump().items():
            setattr(scheduler, field, value)
        return scheduler