# TODO: Implement text-to-image generation using OpenVINO





#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import openvino_genai
from PIL import Image




def generate_image(model_dir: str, prompt: str, device: str = 'CPU') -> Image.Image:
    """Generate an image from text using OpenVINO text-to-image pipeline.
    
    Args:
        model_dir: Path to the model directory
        prompt: Text prompt to generate image from
        device: Device to run on ('CPU' or 'GPU')
    
    Returns:
        PIL.Image.Image: Generated image
    """
    pipe = openvino_genai.Text2ImagePipeline(model_dir, device)

    image_tensor = pipe.generate(
        prompt,
        width=512,
        height=512, 
        num_inference_steps=20,
        num_images_per_prompt=1)

    return Image.fromarray(image_tensor.data[0])


if '__main__' == __name__:
    # Example usage
    image = generate_image("path/to/model", "a scenic landscape")
    image.save("image.bmp")