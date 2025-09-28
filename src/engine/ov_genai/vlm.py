import asyncio
import base64
import gc
import json
import logging
from io import BytesIO
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino as ov
from openvino_genai import (
    GenerationConfig,
    VLMPipeline,
)
from PIL import Image
from transformers import AutoProcessor

from src.server.models.ov_genai import OVGenAI_GenConfig
from src.server.model_registry import ModelLoadConfig, ModelRegistry
from src.engine.ov_genai.streamers import ChunkStreamer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OVGenAI_VLM:
    def __init__(self, load_config: ModelLoadConfig):
        self.model_path
        self.processor: Optional[AutoProcessor] = None
        self.load_config = load_config

    def prepare_inputs(self, messages: List[Dict[str, Any]]) -> Tuple[str, Optional[Union[ov.Tensor, List[ov.Tensor]]]]:
        """
        Convert chat-style messages (with optional raw base64 images in data URLs) into:
        - a prompt string (using model tokenizer chat template on text-only content)
        - a list of ov.Tensor images (one per image), or None if no images
        
        These match VLMPipeline.generate() expected inputs.
        """
        images: List[Image.Image] = []
        text_conversation: List[Dict[str, Any]] = []

        for message in messages:
            # Check if the message content is a list (multimodal content)
            if isinstance(message.get("content", ""), list):
                text_parts: List[str] = []
                for content_item in message["content"]:
                    # Check if this is an image content item
                    if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                        image_url = content_item.get("image_url", {})
                        # Check if it's a base64 encoded image
                        if isinstance(image_url, dict) and isinstance(image_url.get("url", ""), str) and image_url["url"].startswith("data:image/"):
                            # Extract the base64 data
                            base64_data = image_url["url"].split(",", 1)
                            if len(base64_data) > 1:
                                # Decode base64 to binary
                                image_data = base64.b64decode(base64_data[1])
                                # Convert to PIL Image and ensure RGB
                                image = Image.open(BytesIO(image_data)).convert("RGB")
                                images.append(image)
                    # If it's a text content item
                    elif isinstance(content_item, dict) and content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))

                # Create a new message with just the text parts
                text_message = message.copy()
                text_message["content"] = " ".join([t for t in text_parts if isinstance(t, str)]) if text_parts else ""
                text_conversation.append(text_message)
            else:
                text_conversation.append(message)

        # Build prompt from text-only conversation using GenAI tokenizer's chat template
        tokenizer = self.model_path.get_tokenizer()
        prompt: str = tokenizer.apply_chat_template(text_conversation, add_generation_prompt=True)

        # Convert PIL images to ov.Tensor(s). If none, return None for images.
        ov_images: Optional[Union[ov.Tensor, List[ov.Tensor]]] = None
        if images:
            # Pass raw HWC uint8 arrays; VLMPipeline will handle model-specific preprocessing.
            ov_images = [ov.Tensor(np.array(img, dtype=np.uint8)) for img in images]

        return prompt, ov_images

    def generate_type(self, gen_config: OVGenAI_GenConfig):
        """
        Unified generation method that routes to streaming or non-streaming
        based on the stream flag in gen_config. Both paths return an async iterator.
        """
        if gen_config.stream:
            return self.generate_stream(gen_config)
        else:
            return self.generate_text(gen_config)

    async def generate_text(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """
        Async non-streaming generation for VLM.
        Yields in order: metrics (dict), new_text (str).
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        prompt, ov_images = self.prepare_inputs(gen_config.messages)
        if ov_images is not None:
            result = await asyncio.to_thread(self.model_path.generate, prompt, ov_images, generation_kwargs)
        else:
            result = await asyncio.to_thread(self.model_path.generate, prompt, generation_config=generation_kwargs)

        perf_metrics = result.perf_metrics

        text = result.texts[0] if getattr(result, "texts", None) else ""

        metrics_dict = self.collect_metrics(gen_config, perf_metrics)
        yield metrics_dict
        yield text

    async def generate_stream(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Async streaming generation for VLM.
        Yields token chunks (str) as they arrive, then metrics (dict).
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        decoder_tokenizer = self.model_path.get_tokenizer()
        streamer = ChunkStreamer(decoder_tokenizer, gen_config)
        prompt, ov_images = self.prepare_inputs(gen_config.messages)

        async def _run_generation():
            if ov_images is not None:
                return await asyncio.to_thread(
                    self.model_path.generate,
                    prompt,
                    ov_images,
                    generation_kwargs,
                    streamer
                )
            else:
                return await asyncio.to_thread(
                    self.model_path.generate,
                    prompt,
                    generation_config=generation_kwargs,
                    streamer=streamer
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
        """
        ttft_seconds = perf_metrics.get_ttft().mean / 1000
        input_tokens = perf_metrics.get_num_input_tokens()
        prefill_throughput = round(input_tokens / ttft_seconds, 2) if ttft_seconds > 0 else 0

        metrics: Dict[str, Any] = {
            "load_time (s)": round(perf_metrics.get_load_time() / 1000, 2),
            "ttft (s)": round(perf_metrics.get_ttft().mean / 1000, 2),
            "tpot (ms)": round(perf_metrics.get_tpot().mean, 5),
            "prefill_throughput (tokens/s)": prefill_throughput,
            "decode_throughput (tokens/s)": round(perf_metrics.get_throughput().mean, 5),
            "decode_duration (s)": round(perf_metrics.get_generate_duration().mean / 1000, 5),
            "input_token": input_tokens,
            "new_token": perf_metrics.get_num_generated_tokens(),
            "total_token": input_tokens + perf_metrics.get_num_generated_tokens(),
            "stream": gen_config.stream,
        }
        if gen_config.stream and hasattr(gen_config, "stream_chunk_tokens"):
            metrics["stream_chunk_tokens"] = gen_config.stream_chunk_tokens
        return metrics

    def load_model(self, loader: ModelLoadConfig):
        """
        Load model using a ModelLoadConfig configuration and cache the AutoProcessor.
        """

        self.model_path = VLMPipeline(
            loader.model_path,
            loader.device,
            **(loader.runtime_config or {})
        )

        self.processor = AutoProcessor.from_pretrained(loader.model_path)
        logger.info(f"Model loaded successfully: {loader.model_name}")

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """
        Unregister model from registry and free memory resources.
        """
        removed = await registry.register_unload(model_name)

        if self.model_path is not None:
            del self.model_path
            self.model_path = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()
        logger.info(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed


