

import asyncio
import gc
import logging
from typing import Any, AsyncIterator, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer
from optimum.intel import OVModelForFeatureExtraction

from src.server.models.optimum import TokenizerConfig

from typing import Any, AsyncIterator, Dict, Optional

from src.server.model_registry import ModelLoadConfig, ModelRegistry




class Optimum_EMB:
    
    def __init__(self, load_config: ModelLoadConfig):
        self.model_path = None
        self.encoder_tokenizer = None
        self.load_config = load_config
    
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def generate_type(self, tok_config: TokenizerConfig):
        """
        Unified text generation method that routes to streaming or non-streaming
        based on the stream flag in gen_config. Both paths return an async iterator.
        
        Args:
            gen_config: Configuration containing the stream flag and other parameters
            
        Returns:
            - Non-streaming: async iterator yielding [metrics: dict, new_text: str]
            - Streaming: async iterator yielding token chunks (str)... then [metrics: dict, new_text: str]
        """
        return self.generate_embeddings(tok_config)

    def prepare_inputs():
        pass

    async def generate_embeddings(self, tok_config: TokenizerConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        
        # Tokenize the input texts
        batch_dict = self.encoder_tokenizer(
            text=tok_config.text,
            text_pair=tok_config.text_pair,
            text_target=tok_config.text_target,
            text_pair_target=tok_config.text_pair_target,
            add_special_tokens=tok_config.add_special_tokens,
            padding=tok_config.padding,
            truncation=tok_config.truncation,
            max_length=tok_config.max_length,
            stride=tok_config.stride,
            is_split_into_words=tok_config.is_split_into_words,
            pad_to_multiple_of=tok_config.pad_to_multiple_of,
            padding_side=tok_config.padding_side,
            return_tensors=tok_config.return_tensors,
            return_token_type_ids=tok_config.return_token_type_ids,
            return_attention_mask=tok_config.return_attention_mask,
            return_overflowing_tokens=tok_config.return_overflowing_tokens,
            return_special_tokens_mask=tok_config.return_special_tokens_mask,
            return_offsets_mapping=tok_config.return_offsets_mapping,
            return_length=tok_config.return_length,
            verbose=tok_config.verbose
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        # normalize embeddings
        if tok_config.return_tensors=="pt":
            embeddings = F.normalize(embeddings, p=2, dim=1)
        yield embeddings.tolist()

    async def generate_stream():
        pass

    def collect_metrics(self, tok_config: TokenizerConfig, perf_metrics) -> Dict[str, Any]:
        pass

    def load_model(self, loader: ModelLoadConfig):
        """Load model using a ModelLoadConfig configuration and cache the tokenizer.

        Args:
            loader: ModelLoadConfig containing model_path, device, engine, and runtime_config.
        """

        self.model = OVModelForFeatureExtraction.from_pretrained(loader.model_path, 
            device=loader.device, 
            export=False)

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(loader.model_path)
        logging.info(f"Model loaded successfully: {loader.model_name}")

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister model from registry and free memory resources.

        Args:
            registry: ModelRegistry to unregister from
            model_id: Private model identifier returned by register_load

        Returns:
            True if the model was found and unregistered, else False.
        """
        removed = await registry.register_unload(model_name)

        if self.model is not None:
            del self.model
            self.model = None
        
        if self.encoder_tokenizer is not None:
            del self.encoder_tokenizer
            self.encoder_tokenizer = None
        
        gc.collect()
        logging.info(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed