from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional

import openvino as ov
import openvino_genai as genai
from transformers import AutoTokenizer, BatchEncoding

from src.engine.ov_genai.continuous_batching.cb_models import ContinuousBatchConfig
from src.server.model_registry import ModelRegistry
from src.server.models.ov_genai import OVGenAI_GenConfig
from src.server.models.registration import ModelLoadConfig
from src.server.utils.chat import flatten_messages

logger = logging.getLogger(__name__)


class ArcCBLLM:
    """OpenArc adapter for OpenVINO GenAI continuous batching text models."""

    def __init__(self, load_config: ModelLoadConfig):
        self.load_config = load_config
        self.model: genai.ContinuousBatchingPipeline | None = None
        self.encoder_tokenizer = None

    def load_model(self, loader: ModelLoadConfig) -> None:
        """Load a ContinuousBatchingPipeline using the registry load contract."""

        logger.info("%s loading continuous batching pipeline...", loader.model_name)
        logger.info("%s on %s with %s", loader.model_type, loader.device, loader.runtime_config)

        runtime_config = dict(loader.runtime_config or {})
        scheduler = self._build_scheduler_config(runtime_config)

        self.model = genai.ContinuousBatchingPipeline(
            loader.model_path,
            scheduler_config=scheduler,
            device=loader.device,
            properties=runtime_config,
            tokenizer_properties={},
            vision_encoder_properties={},
        )
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(loader.model_path)

        logger.info("%s loaded successfully", loader.model_name)

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister the model and release pipeline resources."""

        removed = await registry.register_unload(model_name)

        if self.model is not None:
            del self.model
            self.model = None

        if self.encoder_tokenizer is not None:
            del self.encoder_tokenizer
            self.encoder_tokenizer = None

        gc.collect()
        logger.info("[%s] unloaded successfully", self.load_config.model_name)
        return removed

    def prepare_inputs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ov.Tensor:
        """
        Convert chat messages into an input_ids tensor using the cached AutoTokenizer.
        """
        if self.encoder_tokenizer is None:
            raise RuntimeError("AutoTokenizer is not loaded")

        prompt_token_ids = self.encoder_tokenizer.apply_chat_template(
            flatten_messages(messages),
            tools=tools,
            add_generation_prompt=True,
            skip_special_tokens=True,
            return_tensors="np",
        )
        if isinstance(prompt_token_ids, BatchEncoding):
            prompt_token_ids = prompt_token_ids["input_ids"]
        return ov.Tensor(prompt_token_ids)

    def add_request(self, request_id: int, gen_config: OVGenAI_GenConfig):
        """
        Add one LLM request through the input_ids ContinuousBatchingPipeline overload.

        Returns:
            (handle, n_input_tokens) so the daemon can do per-request token accounting.
        """
        if self.model is None:
            raise RuntimeError("Continuous batching pipeline is not loaded")

        request_input = self.prepare_inputs(gen_config.messages, gen_config.tools)
        n_input_tokens = int(request_input.shape[-1])
        generation_config = self.create_generation_config(gen_config)
        handle = self.model.add_request(request_id, request_input, generation_config)
        return handle, n_input_tokens

    def step(self) -> None:
        """Advance the continuous batching scheduler by one step."""
        if self.model is None:
            raise RuntimeError("Continuous batching pipeline is not loaded")
        self.model.step()

    def has_non_finished_requests(self) -> bool:
        if self.model is None:
            return False
        return self.model.has_non_finished_requests()

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids with the pipeline's own tokenizer (required for CB streaming)."""
        if self.model is None:
            raise RuntimeError("Continuous batching pipeline is not loaded")
        decoded = self.model.get_tokenizer().decode(token_ids)
        if isinstance(decoded, list):
            return decoded[0] if decoded else ""
        return decoded

    def create_generation_config(self, config: OVGenAI_GenConfig) -> genai.GenerationConfig:
        generation_config = self.model.get_config() if self.model else genai.GenerationConfig()
        generation_config.max_new_tokens = config.max_tokens
        generation_config.temperature = config.temperature
        generation_config.top_k = config.top_k
        generation_config.top_p = config.top_p
        generation_config.repetition_penalty = config.repetition_penalty

        if config.seed:
            generation_config.rng_seed = config.seed
        if config.frequency_penalty:
            generation_config.frequency_penalty = config.frequency_penalty
        if config.presence_penalty:
            generation_config.presence_penalty = config.presence_penalty
        return generation_config

    def collect_metrics(self, input_token: int, new_token: int) -> Dict[str, Any]:
        """
        Per-request token accounting for one streamed CB request.

        The route (/v1/chat/completions) reads `input_token`, `new_token`,
        `total_token` to populate OpenAI `usage`. PipelineMetrics is
        process-level (not per-request) so it is only attached as extra
        informational fields and never substitutes the per-request counts.
        """
        metrics_dict: Dict[str, Any] = {
            "input_token": int(input_token),
            "new_token": int(new_token),
            "total_token": int(input_token) + int(new_token),
            "stream": True,
            "backend": "cb",
        }

        if self.model is not None:
            try:
                pm = self.model.get_metrics()
                metrics_dict["cb_pipeline"] = {
                    "requests": pm.requests,
                    "scheduled_requests": pm.scheduled_requests,
                    "cache_usage": pm.cache_usage,
                    "max_cache_usage": pm.max_cache_usage,
                    "avg_cache_usage": pm.avg_cache_usage,
                }
            except Exception:
                pass

        return metrics_dict

    def _build_scheduler_config(self, runtime_config: dict[str, Any]) -> genai.SchedulerConfig:
        scheduler_values = {
            **{
                key: runtime_config[key]
                for key in ContinuousBatchConfig.model_fields
                if key in runtime_config
            },
            **runtime_config.get("scheduler_config", {}),
            **runtime_config.get("scheduler", {}),
        }
        cb_config = ContinuousBatchConfig(**scheduler_values)

        scheduler = genai.SchedulerConfig()
        scheduler.max_num_batched_tokens = cb_config.max_num_batched_tokens
        scheduler.max_num_seqs = cb_config.max_num_seqs
        scheduler.cache_size = cb_config.cache_size
        scheduler.dynamic_split_fuse = cb_config.dynamic_split_fuse
        scheduler.enable_prefix_caching = cb_config.enable_prefix_caching
        scheduler.use_cache_eviction = cb_config.use_cache_eviction
        return scheduler
