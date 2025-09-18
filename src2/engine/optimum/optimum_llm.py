

import logging

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

from src2.api.model_registry import ModelLoadConfig

import gc
import time
import traceback
import logging
from threading import Thread
from typing import Any, AsyncIterator, Dict, Optional

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer




class Optimum_LLM:
    def __init__():

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

    def load_model():
        pass

    async def unload_model():
        pass