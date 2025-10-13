

import asyncio
import gc
import logging
from typing import Any, AsyncIterator, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM

from src.server.models.optimum import RerankerConfig

from typing import Any, AsyncIterator, Dict

from src.server.model_registry import ModelLoadConfig, ModelRegistry

class Optimum_RR:
    
    def __init__(self, load_config: ModelLoadConfig):
        self.model_path = None
        self.encoder_tokenizer = None
        self.load_config = load_config

    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = "Given a search query, retrieve relevant passages that answer the query"
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)
        return output

    async def generate_rerankings(self, rr_config: RerankerConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        prefix_tokens = self.tokenizer.encode(rr_config.prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(rr_config.suffix, add_special_tokens=False)
        self.max_length = rr_config.max_length
        pairs = [self.format_instruction(rr_config.task, rr_config.query, doc) for doc in rr_config.documents]
        print(pairs)
        # Currently hard coding tokenizer args.  If these are model independent than it is fine.  Otherwise
        # implement the rr_config PreTrainedTokenizerConfig args.
        max_length = 8192
        inputs = self.tokenizer(
            pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )

        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens

        # Currently hard coding tokenizer args.  If these are model independent than it is fine.  Otherwise
        # implement the rr_config PreTrainedTokenizerConfig args.
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        scores = self.compute_logits(inputs)

        ranked_documents = [{"doc":doc, "score":score} for score, doc in sorted(zip(scores,  rr_config.documents), reverse=True)]

        yield ranked_documents

    #not implemented
    def collect_metrics(self, rr_config: RerankerConfig, perf_metrics) -> Dict[str, Any]:
        pass

    def load_model(self, loader: ModelLoadConfig):
        """Load model using a ModelLoadConfig configuration and cache the tokenizer.

        Args:
            loader: ModelLoadConfig containing model_path, device, engine, and runtime_config.
        """

        self.model = OVModelForCausalLM.from_pretrained(loader.model_path, 
            device=loader.device, 
            export=False,
            use_cache=False)

        self.tokenizer = AutoTokenizer.from_pretrained(loader.model_path)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
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
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        gc.collect()
        logging.info(f"[{self.load_config.model_name}] weights and tokenizer unloaded and memory cleaned up")
        return removed