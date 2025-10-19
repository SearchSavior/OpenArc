#!/usr/bin/env python3
"""
End-to-end multimodal chat preprocessing pipeline for OpenVINO GenAI VLMPipeline.

This script:
1. Parses a chat-style message list containing base64-encoded image data.
2. Converts each embedded image into a PIL.Image (and later an ov.Tensor).
3. Inserts appropriate image tokens into the conversation sequence based on model type.
4. Builds the textual prompt using AutoTokenizer.apply_chat_template().
5. Passes both text and image tensors to VLMPipeline.generate() for inference.

Author: <Your Name>
Date: 2025-10-13
"""

# --- Imports -----------------------------------------------------------------
from __future__ import annotations

import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple

from enum import Enum
from PIL import Image
import numpy as np

from transformers import AutoTokenizer
from openvino_genai import VLMPipeline, GenerationConfig
import openvino as ov


# --- Vision Token Mapping ----------------------------------------------------
class VisionToken(Enum):
    """Defines image token syntax for supported Vision-Language Models (VLMs)."""

    INTERNVL2 = "<image>"
    LLAVA_1_5 = "<image>"
    LLAVA_NEXT = "<image>"
    MINICPM_V_2_6 = "(<image>./</image>)"
    PHI_3_VISION = "<|image_{i}|>"
    PHI_4_MM_INSTRUCT = "<|image_{i}|>"
    QWEN2_VL = "<|vision_start|><|image_pad|><|vision_end|>"
    QWEN2_5_VL = "<|vision_start|><|image_pad|><|vision_end|>"
    GEMMA_3 = "<start_of_image>"

    def add_vlm_token(self, index: int) -> str:
        """
        Returns the correctly add_vlm_tokented image token for the given index.

        For models like Phi-3/4 that use numbered tokens, e.g. <|image_1|>,
        this method substitutes the index placeholder. For models that
        use a fixed tag, the raw string is returned.
        """
        if "{i}" in self.value:
            return self.value.add_vlm_token(i=index)
        return self.value


# --- Input Preparation -------------------------------------------------------
def prepare_inputs(
    messages: List[Dict[str, Any]],
    vision_token: VisionToken,
) -> Tuple[str, List[ov.Tensor]]:
    """
    Parse a messages list and prepare text prompt + image tensors for VLM inference.

    Args:
        messages: list of messages, optionally containing multimodal content
        vision_token: VisionToken enum defining the model's image tag syntax

    Returns:
        (tokenized_messages, ov_images)
    """

    images: List[Image.Image] = []
    text_messages: List[Dict[str, Any]] = []

    # Step 1: Extract text and images
    for idx, message in enumerate(messages):
        # Multimodal message (list of dict content items)
        if isinstance(message.get("content", ""), list):
            text_parts: List[str] = []

            for content_item in message["content"]:
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "image_url"
                ):
                    image_url = content_item.get("image_url", {})
                    # Check for embedded base64 data
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

                            # Insert model-specific image token where this image appears
                            token_str = vision_token.add_vlm_token(len(images) - 1)
                            text_parts.append(f" {token_str} ")

                # Handle text segments
                elif isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_parts.append(content_item.get("text", ""))

            # Combine extracted text back into a unified string
            text_message = message.copy()
            text_message["content"] = (
                " ".join([t for t in text_parts if isinstance(t, str)]) if text_parts else ""
            )
            text_messages.append(text_message)

        # Simple text-only message
        else:
            text_messages.append(message)

    # Step 2: Build the chat template prompt
    tokenizer = AutoTokenizer.from_pretrained("/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen2.5-VL-3B-Instruct-int4_sym-ov")
    special_tokens = [v.value for v in VisionToken]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # `apply_chat_template` constructs the conversation in the model's native chat add_vlm_token
    tokenized_messages: str = tokenizer.apply_chat_template(
        text_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Step 3: Convert images to OpenVINO Tensors
    ov_images: List[ov.Tensor] = []
    for img in images:
        arr = np.array(img, dtype=np.uint8)
        tensor = ov.Tensor(arr)
        ov_images.append(tensor)

    return tokenized_messages, ov_images


# --- Entrypoint --------------------------------------------------------------
def main():
    """Demonstration of multimodal input processing and generation."""

    # Load and encode the image for the example
    image_path = "/home/echo/Projects/OpenArc/src/tests/dedication.png"
    with open(image_path, "rb") as f:
        image_data = f.read()
    base64_data = base64.b64encode(image_data).decode("utf-8")
    image_url = f"data:image/png;base64,{base64_data}"

    # Example multimodal conversation structure
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the following image:"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        },
        {
            "role": "assistant",
            "content": "It appears to show a cat sitting on grass."
        },
        {
            "role": "user",
            "content": "What breed is this cat?"
        }
    ]

    # Prepare text prompt and image tensors
    prompt, ov_images = prepare_inputs(messages, VisionToken.QWEN2_5_VL)

    print("=== Generated Prompt ===")
    print(prompt)
    print("========================")

    # Initialize VLM pipeline and generation config
    model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen2.5-VL-3B-Instruct-int4_sym-ov"
    vlm = VLMPipeline(models_path=model_path, device="GPU.2")

    # Perform multimodal generation
    results = vlm.generate(
        prompt=prompt,
        images=ov_images
    )

    print("\n=== Model Output ===")
    print(results)


# --- Main Guard --------------------------------------------------------------
if __name__ == "__main__":
    main()
