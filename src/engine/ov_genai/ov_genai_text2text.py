import gc
import time
import traceback
import logging
from threading import Thread
from typing import Any, AsyncIterator, Dict, Optional

import openvino as ov
from openvino_genai import (
    LLMPipeline,
    DecodedResults,
    EncodedResults,
    GenerationConfig,
    GenerationResult,
    GenerationStatus,
)

class OVGenAI_Text2TextCore:
    pass



    
    #def util_unload_model(self):
    #    """Unload model and free memory"""
    #    del self.model
    #    self.model = None
        
    #    del self.tokenizer
    #    self.tokenizer = None
        
    #    gc.collect()
    #    print("Model unloaded and memory cleaned up")








