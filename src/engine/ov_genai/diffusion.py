import asyncio
import base64
import gc
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional, Union

import numpy as np
from io import BytesIO
from PIL import Image

from openvino_genai import ImageGenerationPerfMetrics, Text2ImagePipeline

from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import ModelLoadConfig
from src.server.schemas.modeling.contract_ovgenai_diffusion import OVGenAI_DiffusionGenConfig

logger = logging.getLogger(__name__)


class GenerationCancelled(ValueError):
    """A denoise loop was interrupted by a cancellation request.

    Subclasses ValueError, the project's request-level error shape, so a cancelled
    generation reports to the one client that asked and never asks the registry to
    unload the model.
    """


class OVGenAI_Diffusion:
    def __init__(self, load_config: ModelLoadConfig):
        self.load_config = load_config
        self.diffusion_model = None
        self._active_request_id: Optional[str] = None
        self._cancel_requested: bool = False

    async def generate_image(self, gen_config: OVGenAI_DiffusionGenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """
        Run text-to-image generation and return base64-encoded PNG images with metrics.

        Yields in order: metrics (dict), then one base64 PNG string per image.
        """
        generate_kwargs = {}
        if gen_config.negative_prompt:
            generate_kwargs["negative_prompt"] = gen_config.negative_prompt
        if gen_config.guidance_scale is not None:
            generate_kwargs["guidance_scale"] = gen_config.guidance_scale
        if gen_config.height is not None:
            generate_kwargs["height"] = gen_config.height
        if gen_config.width is not None:
            generate_kwargs["width"] = gen_config.width
        if gen_config.num_inference_steps is not None:
            generate_kwargs["num_inference_steps"] = gen_config.num_inference_steps
        if gen_config.num_images_per_prompt is not None:
            generate_kwargs["num_images_per_prompt"] = gen_config.num_images_per_prompt
        generate_kwargs["rng_seed"] = (
            gen_config.seed if gen_config.seed is not None
            else int(np.random.randint(0, 2**31 - 1))
        )
        if gen_config.max_sequence_length is not None:
            generate_kwargs["max_sequence_length"] = gen_config.max_sequence_length

        # Track the in-flight request and wire the interrupt hook. Denoising takes tens
        # of seconds and yields nothing on the wire, so cancel() is the only way to stop
        # it, and it acts through this callback between steps.
        self._active_request_id = gen_config.request_id
        self._cancel_requested = False
        generate_kwargs["callback"] = self._interrupt_requested

        try:
            start_time = time.perf_counter()
            result = await asyncio.to_thread(
                self.diffusion_model.generate,
                gen_config.prompt,
                **generate_kwargs,
            )
            wall_time_s = time.perf_counter() - start_time

            if self._cancel_requested:
                # The callback interrupted the loop; generate() hands back an empty
                # tensor, so surface the interruption rather than a blank image.
                raise GenerationCancelled(
                    f"Generation for request {gen_config.request_id} was cancelled mid-denoise"
                )

            pixels = result.data if hasattr(result, "data") else result
            if isinstance(pixels, np.ndarray) and pixels.ndim == 4:
                images = list(pixels)
            elif isinstance(pixels, np.ndarray) and pixels.ndim == 3:
                images = [pixels]
            else:
                # An empty or oddly shaped tensor means nothing was denoised. Indexing it
                # would either throw deep inside PIL or hand the client a blank image.
                raise ValueError(
                    f"Diffusion pipeline returned no image tensor for request {gen_config.request_id}"
                )
            images = [img for img in images if getattr(img, "size", 1)]
            if not images:
                raise ValueError(
                    f"Diffusion pipeline returned empty image tensors for request {gen_config.request_id}"
                )

            # generate() hands back a bare ov.Tensor, so the numbers are read back off
            # the pipeline; the wall clock covers a pipeline that refuses to report them.
            metrics_dict = self.collect_metrics(
                gen_config, self._perf_metrics(), wall_time_s, images
            )

            yield metrics_dict

            for img in images:
                png_bytes = self._encode_png(img)
                yield base64.b64encode(png_bytes).decode("ascii")
        finally:
            self._active_request_id = None
            self._cancel_requested = False

    def _interrupt_requested(self, step, num_steps, latent) -> bool:
        """Pipeline callback hook: True asks generate() to stop the denoise loop."""
        return self._cancel_requested

    async def cancel(self, request_id: str) -> bool:
        """
        Interrupt an in-flight generation by request_id.

        Args:
            request_id: The request ID to cancel

        Returns:
            True if cancellation was triggered, False if request_id didn't match
            the generation currently in flight.
        """
        if request_id is not None and self._active_request_id == request_id:
            self._cancel_requested = True
            logger.info(f"[{self.load_config.model_name}] Cancellation requested for request {request_id}")
            return True
        return False

    def _encode_png(self, image_array) -> bytes:
        """Encode a diffusion image tensor/array to PNG bytes.

        Accepts uint8 or float pixel arrays in `(H, W, C)` layout, and also transposes
        the common `(C, H, W)` layout to HWC. Float pixels in `0..1` are scaled by 255;
        float/int pixels already scaled around `0..255` are clipped instead.
        """
        arr = np.asarray(image_array)

        if arr.ndim != 3:
            raise ValueError("Diffusion pipeline returned an unsupported image array shape")

        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[-1] not in (1, 3, 4):
            raise ValueError("Diffusion pipeline returned an image array with an invalid channel count")

        if arr.size == 0:
            raise ValueError("Diffusion pipeline returned an empty image array")

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float64)
                min_value = float(arr.min())
                max_value = float(arr.max())

                if 0.0 <= min_value and max_value <= 1.0:
                    arr = arr * 255.0
                else:
                    arr = np.clip(arr, 0.0, 255.0)
            else:
                arr = np.clip(arr.astype(np.float64), 0.0, 255.0)

            arr = arr.astype(np.uint8)

        if arr.shape[-1] == 1:
            img = Image.fromarray(arr[..., 0], mode="L")
        else:
            img = Image.fromarray(arr)

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _perf_metrics(self) -> Optional[ImageGenerationPerfMetrics]:
        """Stats for the generate() call that just finished, or None if unavailable.

        Text2ImagePipeline.generate hands back a bare ov.Tensor, so the numbers hang
        on the pipeline instead of the result. The raw per-step lists are reset on each
        call, which keeps this snapshot about the request that just ran rather than
        everything the model has ever done.
        """
        try:
            return self.diffusion_model.get_performance_metrics()
        except Exception:
            logger.debug("Diffusion perf metrics unavailable", exc_info=True)
            return None

    @staticmethod
    def _ms(value) -> Optional[float]:
        """Normalize an ImageGenerationPerfMetrics getter to milliseconds, or None.

        The getters are heterogeneous: totals come back as floats, per-component
        averages as MeanStdPair (mean, std), text encoders as one {name: ms} dict.
        A component the architecture does not have (no transformer in SD1.5, no UNet
        in SD3/Flux) reports 0.0, which becomes None so the client only sees the parts
        that actually ran.
        """
        if isinstance(value, dict):
            value = sum(value.values())
        mean = getattr(value, "mean", value)
        try:
            ms = float(mean)
        except (TypeError, ValueError):
            return None
        return ms if ms > 0 else None

    def collect_metrics(
        self,
        gen_config: OVGenAI_DiffusionGenConfig,
        perf_metrics: Optional[ImageGenerationPerfMetrics],
        wall_time_s: float,
        images: list,
    ) -> Dict[str, Any]:
        """Collect key performance metrics for one text-to-image generation.

        Args:
            gen_config: the request's OVGenAI_DiffusionGenConfig
            perf_metrics: pipeline stats for this call, or None when unavailable
            wall_time_s: generate() wall time, the fallback duration
            images: decoded image arrays, for the reported image size

        Returns:
            metrics: Dict[str, Any]
        """
        duration_ms = self._ms(perf_metrics.get_generate_duration()) if perf_metrics else None
        duration_s = duration_ms / 1000 if duration_ms is not None else wall_time_s

        steps = gen_config.num_inference_steps
        if steps is None and perf_metrics is not None:
            # Unset num_inference_steps leaves the scheduler's own default, so read it
            # back off the per-step timings: the loop logs one entry per step plus one
            # for the trailing decode pass.
            steps = max(len(perf_metrics.raw_metrics.iteration_durations) - 1, 0)

        metrics: Dict[str, Any] = {
            "steps": steps,
            "images": len(images),
            "generate_duration (s)": round(duration_s, 4),
            "step_time (s)": round(duration_s / steps, 4) if steps else None,
            "throughput (it/s)": round(steps / duration_s, 2) if steps and duration_s else None,
        }

        if perf_metrics is not None:
            load_ms = self._ms(perf_metrics.get_load_time())
            if load_ms is not None:
                metrics["load_time (s)"] = round(load_ms / 1000, 4)
            # Per-component split, so the client can see which stage of the pipeline
            # the device is actually spending the request in.
            for key, getter in (
                ("text_encoder (s)", perf_metrics.get_text_encoder_infer_duration),
                ("vae_encoder (s)", perf_metrics.get_vae_encoder_infer_duration),
                ("vae_decoder (s)", perf_metrics.get_vae_decoder_infer_duration),
                ("unet (s)", perf_metrics.get_unet_infer_duration),
                ("transformer (s)", perf_metrics.get_transformer_infer_duration),
            ):
                component_ms = self._ms(getter())
                if component_ms is not None:
                    metrics[key] = round(component_ms / 1000, 4)

        if images and images[0].ndim >= 2:
            metrics["height"] = int(images[0].shape[0])
            metrics["width"] = int(images[0].shape[1])

        # A metric with no number behind it (steps unknown, a component the architecture
        # does not have) is noise in the response body, so drop it rather than send null.
        return {key: value for key, value in metrics.items() if value is not None}

    def load_model(self, loader: ModelLoadConfig) -> None:
        """Load a diffusion model into a Text2ImagePipeline for the given device."""
        pipeline_kwargs = {**(loader.runtime_config or {})}
        if loader.cache_dir:
            pipeline_kwargs["CACHE_DIR"] = loader.cache_dir

        self.diffusion_model = Text2ImagePipeline(
            loader.model_path,
            loader.device,
            **pipeline_kwargs,
        )

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister model from registry and free memory resources."""
        removed = await registry.register_unload(model_name)

        if self.diffusion_model is not None:
            del self.diffusion_model
            self.diffusion_model = None

        gc.collect()
        logger.info(f"[{self.load_config.model_name}] diffusion model unloaded and memory cleaned up")
        return removed
