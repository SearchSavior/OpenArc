

import asyncio
import gc
import logging
from typing import Any, AsyncIterator, Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer
from optimum.intel import OVModelForFeatureExtraction

from src.server.schemas.modeling.contract_optimum_emb import PreTrainedTokenizerConfig

from typing import Any, AsyncIterator, Dict

from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import ModelLoadConfig




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

    @staticmethod
    def position_ids_from_mask(attention_mask: Tensor) -> Tensor:
        """Positions 0..n-1 over the real tokens of each row, for left or right padding.

        Same rule optimum-intel's OVModelForCausalLM uses when it builds position_ids.
        """
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def _last_hidden_state(self, batch_dict) -> Tensor:
        input_names = getattr(self.model, "input_names", None) or {}
        if "position_ids" not in input_names:
            return self.model(**batch_dict).last_hidden_state

        # Decoder-based embedding IRs exported before optimum-intel 1.26 (for example
        # Echo9Zulu/Qwen3-Embedding-0.6B-int8_asym-ov) take position_ids as a graph input.
        # OVModelForFeatureExtraction.forward() drops it, so the GPU plugin fails on the
        # missing input and CPU computes on an unset tensor (NaN). Call the compiled model
        # with every input it expects.
        attention_mask = batch_dict["attention_mask"]
        input_ids = batch_dict["input_ids"].cpu().numpy()
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask.cpu().numpy(),
            "position_ids": self.position_ids_from_mask(attention_mask).cpu().numpy(),
        }
        if "token_type_ids" in input_names:  # same fallback as forward()
            token_type_ids = batch_dict.get("token_type_ids")
            inputs["token_type_ids"] = (
                token_type_ids.cpu().numpy() if token_type_ids is not None else np.zeros_like(input_ids)
            )
        if "decoder_input_ids" in input_names:  # same as forward()
            inputs["decoder_input_ids"] = input_ids
        self.model.compile()
        outputs = self.model._inference(inputs)
        return torch.from_numpy(outputs["last_hidden_state"])

    # Cap for truncation when the client gives no max_length: the model's own limit, but no
    # more than 8192 tokens (Qwen3-Embedding's reference max_length) to bound memory per request.
    DEFAULT_MAX_INPUT_TOKENS = 8192

    def _model_max_input_tokens(self) -> int:
        limits = [self.DEFAULT_MAX_INPUT_TOKENS]
        config = getattr(self.model, "config", None)
        max_positions = getattr(config, "max_position_embeddings", None)
        if isinstance(max_positions, int) and max_positions > 0:
            limits.append(max_positions)
        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 1_000_000:  # HF uses ~1e30 for "unset"
            limits.append(tokenizer_max)
        return min(limits)

    def _resolve_max_length(self, tok_config: PreTrainedTokenizerConfig):
        if tok_config.max_length is not None or not tok_config.truncation:
            return tok_config.max_length
        return self._model_max_input_tokens()

    async def generate_embeddings(self, tok_config: PreTrainedTokenizerConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            text=tok_config.text,
            text_pair=tok_config.text_pair,
            text_target=tok_config.text_target,
            text_pair_target=tok_config.text_pair_target,
            add_special_tokens=tok_config.add_special_tokens,
            padding=tok_config.padding,
            truncation=tok_config.truncation,
            max_length=self._resolve_max_length(tok_config),
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
        last_hidden_state = self._last_hidden_state(batch_dict)
        embeddings = self.last_token_pool(last_hidden_state, batch_dict["attention_mask"])
        # normalize embeddings
        if tok_config.return_tensors=="pt":
            embeddings = F.normalize(embeddings, p=2, dim=1)
        input_tokens = int(batch_dict["attention_mask"].sum())
        yield {"input_token": input_tokens, "total_token": input_tokens}
        yield embeddings.tolist()

    def collect_metrics(self, tok_config: PreTrainedTokenizerConfig, perf_metrics) -> Dict[str, Any]:
        pass

    def load_model(self, loader: ModelLoadConfig):
        """Load model using a ModelLoadConfig configuration and cache the tokenizer.

        Args:
            loader: ModelLoadConfig containing model_path, device, engine, and runtime_config.
        """

        ov_config = {**(loader.runtime_config or {})}
        if loader.cache_dir:
            ov_config["CACHE_DIR"] = loader.cache_dir

        from_pretrained_kwargs = {"device": loader.device, "export": False}
        if ov_config:
            from_pretrained_kwargs["ov_config"] = ov_config

        self.model = OVModelForFeatureExtraction.from_pretrained(loader.model_path,
            **from_pretrained_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(loader.model_path)
        self._log_input_handling(loader.model_name)
        logging.info(f"Model loaded successfully: {loader.model_name}")

    # Inputs this engine knows how to fill: forward() covers all but position_ids,
    # which _last_hidden_state() builds from the attention mask.
    KNOWN_INPUTS = {"input_ids", "attention_mask", "token_type_ids", "decoder_input_ids", "position_ids"}

    def _log_input_handling(self, model_name: str) -> None:
        input_names = set(getattr(self.model, "input_names", None) or {})
        if "position_ids" in input_names:
            logging.info(
                f"[{model_name}] IR takes position_ids: building them from the attention mask "
                "(decoder embedding export from optimum-intel < 1.26)"
            )
        unknown = input_names - self.KNOWN_INPUTS
        if unknown:
            logging.warning(
                f"[{model_name}] IR has inputs this engine does not fill: {sorted(unknown)}. "
                "Inference will likely fail; re-export with optimum-intel --task feature-extraction."
            )

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
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        gc.collect()
        logging.info(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed