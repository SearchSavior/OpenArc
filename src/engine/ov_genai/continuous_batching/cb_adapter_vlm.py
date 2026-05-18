from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openvino as ov
from PIL import Image

from src.engine.ov_genai.continuous_batching.cb_adapter_llm import ArcCBLLM
from src.server.models.ov_genai import OVGenAI_GenConfig, VLM_VISION_TOKENS
from src.server.models.registration import ModelLoadConfig
from src.server.utils.chat import flatten_message_content


class ArcCBVLM(ArcCBLLM):
    """OpenArc adapter for OpenVINO GenAI continuous batching multimodal models."""

    def __init__(self, load_config: ModelLoadConfig):
        super().__init__(load_config)
        self.vision_token = None

    def load_model(self, loader: ModelLoadConfig) -> None:
        """Load the shared ContinuousBatchingPipeline and cache VLM token metadata."""

        super().load_model(loader)
        self.vision_token = VLM_VISION_TOKENS.get(loader.vlm_type)
        if self.vision_token is None:
            raise ValueError(
                f"Unknown VLM type: {loader.vlm_type}. Supported: {list(VLM_VISION_TOKENS.keys())}"
            )

    def _vision_token_for_index(self, index: int) -> str:
        token_template = self.vision_token if self.vision_token is not None else ""
        if "{i}" in token_template:
            return token_template.replace("{i}", str(index))
        return token_template

    def prepare_inputs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[ov.Tensor]]:
        """Convert OpenAI-style multimodal chat messages to a prompt and image tensors."""

        if self.encoder_tokenizer is None:
            raise RuntimeError("AutoTokenizer is not loaded")

        images: List[Image.Image] = []
        text_messages: List[Dict[str, Any]] = []

        for message in messages:
            if isinstance(message.get("content", ""), list):
                text_parts: List[str] = []

                for content_item in message["content"]:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "image_url"
                    ):
                        image_url = content_item.get("image_url", {})
                        if (
                            isinstance(image_url, dict)
                            and isinstance(image_url.get("url", ""), str)
                            and image_url["url"].startswith("data:image/")
                        ):
                            base64_data = image_url["url"].split(",", 1)
                            if len(base64_data) > 1:
                                image_data = base64.b64decode(base64_data[1])
                                image = Image.open(BytesIO(image_data)).convert("RGB")
                                images.append(image)
                                token = self._vision_token_for_index(len(images) - 1)
                                text_parts.append(f" {token} ")
                    elif (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "text"
                    ):
                        text_parts.append(content_item.get("text", ""))

                text_message = message.copy()
                text_message["content"] = flatten_message_content(
                    " ".join(text_parts) if text_parts else ""
                )
                text_messages.append(text_message)
            else:
                text_messages.append(
                    {**message, "content": flatten_message_content(message.get("content"))}
                )

        prompt = self.encoder_tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            tools=tools,
            add_generation_prompt=True,
        )
        ov_images = [ov.Tensor(np.array(image, dtype=np.uint8)) for image in images]
        return prompt, ov_images

    def add_request(self, request_id: int, gen_config: OVGenAI_GenConfig):
        """Add one VLM request through the prompt/images ContinuousBatchingPipeline overload."""

        if self.model is None:
            raise RuntimeError("Continuous batching pipeline is not loaded")

        prompt, images = self.prepare_inputs(gen_config.messages, gen_config.tools)
        generation_config = self.create_generation_config(gen_config)
        return self.model.add_request(request_id, prompt, images, generation_config)
