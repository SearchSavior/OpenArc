from __future__ import annotations

import logging

from src.server.models.registration import ModelLoadConfig

logger = logging.getLogger(__name__)


class ArcCBVLM:
    """Reserved adapter for continuous batching vision models. Not implemented."""

    def __init__(self, load_config: ModelLoadConfig):
        self.load_config = load_config

    def load_model(self, loader: ModelLoadConfig) -> None:
        raise NotImplementedError("cb_vlm is not implemented")
