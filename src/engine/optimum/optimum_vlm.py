

import gc
import logging
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

from src.server.model_registry import ModelRegistry
from src.server.models.registration import ModelLoadConfig


class Optimum_VLM:
    def __init__(self, load_config: ModelLoadConfig):
        self.load_config = load_config
        self.model = None
        self.tokenizer = None

    def generate_type():
        pass

    def prepare_inputs():
        pass

    async def generate_text():
        pass

    async def generate_stream():
        pass

    def collect_metrics():
        pass

    def load_model(self, loader: ModelLoadConfig):
        """Load model using a ModelLoadConfig configuration and cache the tokenizer.

        Args:
            loader: ModelLoadConfig containing model_path, device, engine, and runtime_config.
        """

        self.model = OVModelForVisualCausalLM.from_pretrained(
            loader.model_path,
            device=loader.device,
            export=False,
        )

        self.tokenizer = AutoProcessor.from_pretrained(loader.model_path)
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

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        logging.info(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed
