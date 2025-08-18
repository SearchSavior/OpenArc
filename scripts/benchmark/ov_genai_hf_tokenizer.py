import gc
import json
import logging
import queue
import time
import traceback
from pathlib import Path
from threading import Thread
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import openvino_genai as ov_genai
from openvino_genai import GenerationConfig, LLMPipeline, StreamerBase
from pydantic import BaseModel, Field
from collections import namedtuple
from transformers import AutoTokenizer
import openvino as ov


# ---------------------------
# Pydantic Config Models
# ---------------------------
class OVGenAI_LoadConfig(BaseModel):
    """
    Configuration for loading an OpenVINO GenAI model.
    """
    model_path: str = Field(..., description="Path to the model directory (top-level).")
    device: str = Field(default="CPU", description="Target device for inference.")
    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional OpenVINO runtime properties."
    )


class OVGenAI_TextGenConfig(BaseModel):
    """
    Configuration for text generation with an OpenVINO GenAI pipeline.
    """
    conversation: str = Field(
        ...,
        description="Formatted conversation string ready for the model."
    )
    max_new_tokens: int = Field(
        default=50,
        description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature; higher values increase randomness."
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling cutoff."
    )
    top_p: float = Field(
        default=1.0,
        description="Nucleus sampling probability cutoff."
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Penalty for repeating tokens."
    )
    apply_chat_template: bool = Field(
        default=False,
        description="If True, apply the tokenizer's chat template to the conversation."
    )


DecodedResults = namedtuple("DecodedResults", ["perf_metrics", "scores", "texts"])


class LLMPipelineWithHFTokenizer(ov_genai.LLMPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_dir = kwargs["model_dir"] if "model_dir" in kwargs else args[0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def generate(self, *args, **kwargs):
        texts = kwargs.pop("inputs", None)
        if texts is None:
            texts, args = args[0], args[1:]
        if kwargs.pop("apply_chat_template", False):
            inputs = self.tokenizer.apply_chat_template(texts, add_generation_prompt=True, return_tensors="np")
            inputs = ov.Tensor(inputs)
        else:
            inputs = ov.Tensor(self.tokenizer(texts, return_tensors="np")["input_ids"])
        out = super().generate(inputs, *args, **kwargs)
        res = DecodedResults(out.perf_metrics, out.scores, self.tokenizer.batch_decode(out.tokens))
        return res


class OVGenAI_Text2Text:
    def __init__(self):
        self.model = None

    def load_model(self, config: OVGenAI_LoadConfig):
        """
        Loads an OpenVINO GenAI text-to-text model.
        """
        id_model = Path(config.model_path)
        if not id_model.exists():
            raise FileNotFoundError(f"Model path does not exist: {id_model}")

        self.model = LLMPipelineWithHFTokenizer(
            id_model,
            config.device,
            **(config.properties or {})
        )

    def generate_text(self, config: OVGenAI_TextGenConfig) -> str:
        """
        Non-streaming text generation.
        """
        generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty
        )

        result = self.model.generate([config.conversation], generation_config)
        perf_metrics = result.perf_metrics

        metrics_dict = {
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
            'generate_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2),
        }
        return metrics_dict, result.texts[0]

    def generate_stream(self, config: OVGenAI_TextGenConfig):
        """
        Streaming text generation with profiling.
        """
        generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty
        )

        texts = config.conversation
        if config.apply_chat_template:
            try:
                texts = json.loads(config.conversation)
            except Exception:
                pass

        result = self.model.generate(
            texts,
            generation_config,
            apply_chat_template=config.apply_chat_template
        )

        perf_metrics = result.perf_metrics

        metrics_dict = {
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
            'generate_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2)
        }

        final_text = result.texts[0]
        return metrics_dict, final_text


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    load_cfg = OVGenAI_LoadConfig(
        model_path="/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Llama-3.2-3B-Instruct-abliterated-OpenVINO/Llama-3.2-3B-Instruct-abliterated-int4_asym-ov",
        device="GPU.0"
    )

    conversation_messages = [
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am an AI assistant who tells dirty jokes. How can I help you?"},
        {"role": "user", "content": "Tell me a joke."}
    ]
    conversation_json = json.dumps(conversation_messages, indent=2)

    textgeneration_config = OVGenAI_TextGenConfig(
        conversation=conversation_json,
        max_new_tokens=1024,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1
    )

    text_gen = OVGenAI_Text2Text()
    text_gen.load_model(load_cfg)

    metrics, output = text_gen.generate_stream(textgeneration_config)
    print("\n\n=== Streaming Metrics ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== Output ===\n", output)
