import asyncio
import base64
import gc

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
from transformers import AutoTokenizer

from src.server.models.ov_genai import OVGenAI_GenConfig, VLM_VISION_TOKENS
from src.server.utils.chat import flatten_message_content
from src.server.model_registry import ModelRegistry
from src.server.models.registration import ModelLoadConfig
from src.engine.ov_genai.streamers import ChunkStreamer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OVGenAI_VLM:
    def __init__(self, load_config: ModelLoadConfig):
        self.model_path = None
        self.tokenizer = None
        self.vision_token = None
        self.load_config = load_config

    def _vision_token_for_index(self, index: int) -> str:
        """
        Return the correctly formatted vision token for the given image index.
        Handles templates that may contain an index placeholder like '{i}'.
        """
        token_template = self.vision_token if self.vision_token is not None else ""
        if "{i}" in token_template:
            return token_template.replace("{i}", str(index))
        return token_template

    def prepare_inputs(self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
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
                                token_str = self._vision_token_for_index(len(images) - 1)
                                text_parts.append(f" {token_str} ")

                    # Handle text segments
                    elif isinstance(content_item, dict) and content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))

                # Combine extracted text back into a unified string
                text_message = message.copy()
                text_message["content"] = flatten_message_content(
                    " ".join([t for t in text_parts if isinstance(t, str)]) if text_parts else ""
                )
                text_messages.append(text_message)

            # Simple text-only message
            else:
                text_messages.append(
                    {**message, "content": flatten_message_content(message.get("content"))}
                )

        # Step 2: Build the chat template prompt using cached tokenizer
        tokenizer = self.tokenizer
        tokenized_messages: str = tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            tools=tools,
            add_generation_prompt=True
        )

        # Step 3: Convert images to OpenVINO Tensors
        ov_images: List[ov.Tensor] = []
        for img in images:
            arr = np.array(img, dtype=np.uint8)
            tensor = ov.Tensor(arr)
            ov_images.append(tensor)

        return tokenized_messages, ov_images

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
        try:
            generation_kwargs = GenerationConfig(
                max_new_tokens=gen_config.max_tokens,
                temperature=gen_config.temperature,
                top_k=gen_config.top_k,
                top_p=gen_config.top_p,
                repetition_penalty=gen_config.repetition_penalty,
            )

            prompt, ov_images = self.prepare_inputs(gen_config.messages, gen_config.tools)
            
            result = await asyncio.to_thread(
                self.model_path.generate,
                prompt=prompt,
                **({'images': ov_images} if len(ov_images) > 0 else {}),
                generation_config=generation_kwargs,
            )

            perf_metrics = result.perf_metrics

            text = result.texts[0] if getattr(result, "texts", None) else ""
            logger.info(f"[{self.load_config.model_name}] Generation completed, generated {len(text)} characters")

            metrics_dict = self.collect_metrics(gen_config, perf_metrics)
            yield metrics_dict
            yield text
        except Exception as e:
            logger.error(f"[{self.load_config.model_name}] Error during non-streaming generation: {e}", exc_info=True)
            raise

    async def generate_stream(self, 
    gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Async streaming generation for VLM.
        Yields token chunks (str) as they arrive, then metrics (dict).
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        decoder_tokenizer = self.model_path.get_tokenizer()
        streamer = ChunkStreamer(decoder_tokenizer, gen_config)
        prompt, ov_images = self.prepare_inputs(gen_config.messages, gen_config.tools)

        async def _run_generation():
            return await asyncio.to_thread(
                self.model_path.generate,
                prompt=prompt,
                **({'images': ov_images} if len(ov_images) > 0 else {}),
                generation_config=generation_kwargs,
                streamer=streamer,
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
        Load the VLMPipeline and cache the tokenizer and vision token.
        """
        try:
            logger.info(f"{loader.model_type} on {loader.device} with {loader.runtime_config}")
            
            self.model_path = VLMPipeline(
                loader.model_path,
                loader.device,
                **(loader.runtime_config or {})
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(loader.model_path)
    
            # Get vision token from the mapping using vlm_type as key
            self.vision_token = VLM_VISION_TOKENS.get(loader.vlm_type)
            if self.vision_token is None:
                raise ValueError(f"Unknown VLM type: {loader.vlm_type}. Supported: {list(VLM_VISION_TOKENS.keys())}")

            logger.info(f"{loader.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"[{loader.model_name}] Failed to initialize VLMPipeline: {e}", exc_info=True)

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """
        Unregister model from registry and free memory resources.
        """
        removed = await registry.register_unload(model_name)

        if self.model_path is not None:
            del self.model_path
            self.model_path = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if self.vision_token is not None:
            del self.vision_token
            self.vision_token = None

        gc.collect()
        logger.info(f"[{self.load_config.model_name}] unloaded successfully")
        return removed

