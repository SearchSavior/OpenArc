from typing import Optional

from pydantic import BaseModel, Field


class OVGenAI_DiffusionGenConfig(BaseModel):
    """
    Configuration for text-to-image generation with an OpenVINO GenAI diffusion pipeline.
    Supports Stable Diffusion, SDXL, SD3, Flux, and LCM models.
    """
    prompt: Optional[str] = Field(
        default=None,
        description=(
            "Text prompt describing the image to generate. OpenArc accepts an empty prompt; "
            "the route names generated PNGs from the prompt, falling back to empty_prompt.png."
        )
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Negative prompt to discourage certain features."
    )
    guidance_scale: Optional[float] = Field(
        default=None,
        description=(
            "Classifier-free guidance scale. Higher values follow the prompt more closely. "
            "Unset leaves the model's own default: ~7.5 for Stable Diffusion/SDXL, "
            "~3.5-4.5 for SD3 and Flux."
        )
    )
    height: Optional[int] = Field(
        default=None,
        description=(
            "Output image height in pixels. Must be divisible by 8. "
            "Unset leaves the model's native resolution: 512 for Stable Diffusion, "
            "768-1024 for SDXL, 1024 for SD3 and Flux."
        )
    )
    width: Optional[int] = Field(
        default=None,
        description=(
            "Output image width in pixels. Must be divisible by 8. "
            "Unset leaves the model's native resolution; see height."
        )
    )
    num_inference_steps: Optional[int] = Field(
        default=None,
        description=(
            "Number of denoising steps. More steps = higher quality but slower. "
            "Unset leaves the pipeline's own default, 50 for Stable Diffusion/SDXL/SD3/Flux "
            "and ~4 for LCM; the /v1/images/generations metrics report the count that actually ran."
        )
    )
    num_images_per_prompt: Optional[int] = Field(
        default=None,
        description="Number of images to generate per prompt. Defaults to 1 when unset."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Fix the RNG seed for reproducible image generation."
    )
    strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "How far to move away from an initial image during image-to-image generation, "
            "0.0 (barely changes the input) to 1.0 (fully renoise, i.e. plain text-to-image). "
            "Only consumed by an Image2ImagePipeline; rejected by the text-to-image route."
        )
    )
    max_sequence_length: Optional[int] = Field(
        default=None,
        description="Maximum sequence length for the text encoder. Defaults to model-specific value."
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking and cancellation."
    )
