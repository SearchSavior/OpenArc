from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import openvino as ov
import openvino_genai as genai

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

        logger.info("%s loaded successfully", loader.model_name)

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister the model and release pipeline resources."""

        removed = await registry.register_unload(model_name)

        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        logger.info("[%s] unloaded successfully", self.load_config.model_name)
        return removed

    def prepare_inputs(self, gen_config: OVGenAI_GenConfig) -> str | ov.Tensor:
        """Prepare the LLM request payload for ContinuousBatchingPipeline.add_request."""

        if gen_config.input_ids:
            input_ids = np.array(gen_config.input_ids, dtype=np.int64).reshape(1, -1)
            return ov.Tensor(input_ids)

        if gen_config.prompt:
            return gen_config.prompt

        if self.model is None:
            raise RuntimeError("Continuous batching pipeline is not loaded")

        tokenizer = self.model.get_tokenizer()
        return tokenizer.apply_chat_template(
            flatten_messages(gen_config.messages),
            add_generation_prompt=True,
            tools=gen_config.tools,
        )

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
