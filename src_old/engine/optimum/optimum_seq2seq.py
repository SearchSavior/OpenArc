# TODO: Implement support for generating embeddings.


import gc
import time
import traceback
import logging
from threading import Thread
from typing import Any, AsyncIterator, Dict, Optional



from optimum.intel import OVModelForSeq2SeqLM
import torch

from .optimum_base_config import (
    OV_Config,
    OV_GenerationConfig,
    OV_LoadModelConfig,
    ModelType
)

class Optimum_Seq2SeqCore

    def util_unload_model(self):
        """Unload model and free memory"""
        del self.model
        self.model = None
        
        del self.tokenizer
        self.tokenizer = None
        
        gc.collect()
        print("Model unloaded and memory cleaned up")