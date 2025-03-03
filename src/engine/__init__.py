# src/engine/__init__.py
"""
Engine module for OpenArc.

This module contains the core inference engines for different model types:
- optimum: For Optimum-Intel models
- ov_genai: For OpenVINO GenAI models
"""

# You can optionally expose key classes at the module level
from src.engine.optimum.optimum_inference_core import (
    OV_LoadModelConfig,
    OV_Config,
    OV_GenerationConfig,
    Optimum_InferenceCore,
)

# Add version info if desired
__version__ = "0.1.0"