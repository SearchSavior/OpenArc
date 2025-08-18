import gc
import json
import logging
import queue
import time
import traceback
from pathlib import Path
from threading import Thread
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import openvino_genai
from openvino_genai import GenerationConfig, LLMPipeline, StreamerBase
from pydantic import BaseModel, Field


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
    stream_chunk_tokens: Optional[int] = Field(
        default=None,
        description="If set > 1, stream output in chunks of this many tokens using ChunkStreamer."
    )


class IterableStreamer(StreamerBase):
    """
    A custom streamer class for handling token streaming and detokenization with buffering.
    Includes timing instrumentation for profiling overhead sources.
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_cache: List[int] = []
        self.text_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.print_len = 0
        self.decoded_lengths: List[int] = []

        # Profiling accumulators
        self.decode_times: List[float] = []
        self.queue_times: List[float] = []
        self.write_times: List[float] = []

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        return openvino_genai.StreamingStatus.RUNNING

    def write_word(self, word: str):
        start = time.perf_counter()
        self.text_queue.put(word)
        self.queue_times.append((time.perf_counter() - start) * 1000)  # ms

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        start_total = time.perf_counter()

        # Cache tokens
        if type(token) is list:
            self.tokens_cache += token
            self.decoded_lengths += [-2 for _ in range(len(token) - 1)]
        else:
            self.tokens_cache.append(token)

        # Decode timing
        start_decode = time.perf_counter()
        text = self.tokenizer.decode(self.tokens_cache)
        self.decode_times.append((time.perf_counter() - start_decode) * 1000)  # ms
        self.decoded_lengths.append(len(text))

        # Logic for deciding what to emit
        word = ""
        delay_n_tokens = 3
        if len(text) > self.print_len and "\n" == text[-1]:
            word = text[self.print_len:]
            self.tokens_cache = []
            self.decoded_lengths = []
            self.print_len = 0
        elif len(text) > 0 and text[-1] == chr(65533):
            self.decoded_lengths[-1] = -1
        elif len(self.tokens_cache) >= delay_n_tokens:
            self.compute_decoded_length_for_position(len(self.decoded_lengths) - delay_n_tokens)
            print_until = self.decoded_lengths[-delay_n_tokens]
            if print_until != -1 and print_until > self.print_len:
                word = text[self.print_len:print_until]
                self.print_len = print_until
        if word:
            self.write_word(word)

        stop_flag = self.get_stop_flag()
        if stop_flag != openvino_genai.StreamingStatus.RUNNING:
            self.end()

        self.write_times.append((time.perf_counter() - start_total) * 1000)  # ms
        return stop_flag

    def compute_decoded_length_for_position(self, cache_position: int):
        if self.decoded_lengths[cache_position] != -2:
            return
        cache_for_position = self.tokens_cache[:cache_position + 1]

        start_decode = time.perf_counter()
        text_for_position = self.tokenizer.decode(cache_for_position)
        self.decode_times.append((time.perf_counter() - start_decode) * 1000)  # ms

        if len(text_for_position) > 0 and text_for_position[-1] == chr(65533):
            self.decoded_lengths[cache_position] = -1
        else:
            self.decoded_lengths[cache_position] = len(text_for_position)

    def end(self):
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len:]
            if word:
                self.write_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.text_queue.put(None)

    def summarize_timings(self) -> Dict[str, float]:
        """Summarize average timings per operation in ms."""
        return {
            "decode_avg_ms": sum(self.decode_times) / len(self.decode_times) if self.decode_times else 0,
            "queue_avg_ms": sum(self.queue_times) / len(self.queue_times) if self.queue_times else 0,
            "write_avg_ms": sum(self.write_times) / len(self.write_times) if self.write_times else 0,
            "decode_calls": len(self.decode_times),
            "queue_calls": len(self.queue_times),
            "write_calls": len(self.write_times),
        }


class ChunkStreamer(IterableStreamer):
    def __init__(self, tokenizer, tokens_len: int):
        super().__init__(tokenizer)
        self.tokens_len = tokens_len

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        if (len(self.tokens_cache) + 1) % self.tokens_len == 0:
            return super().write(token)
        if type(token) is list:
            self.tokens_cache += token
            self.decoded_lengths += [-2 for _ in range(len(token))]
        else:
            self.tokens_cache.append(token)
            self.decoded_lengths.append(-2)
        return openvino_genai.StreamingStatus.RUNNING


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

        self.model = LLMPipeline(
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

        tokenizer = self.model.get_tokenizer()
        # Use chunked streaming if configured (>1), otherwise default token-by-token streaming.
        stream_tokens = getattr(config, "stream_chunk_tokens", None)
        streamer = (
            ChunkStreamer(tokenizer, stream_tokens) if stream_tokens and stream_tokens > 1 else IterableStreamer(tokenizer)
        )
        collected_chunks: List[str] = []

        def token_collector():
            for word in streamer:
                print(word, end="", flush=True)
                collected_chunks.append(word)

        printer_thread = Thread(target=token_collector, daemon=True)
        printer_thread.start()

        result = self.model.generate([config.conversation], generation_config, streamer)
        printer_thread.join()

        perf_metrics = result.perf_metrics
        timing_summary = streamer.summarize_timings()

        metrics_dict = {
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
            'generate_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2),
            'decode_avg_ms': timing_summary['decode_avg_ms'],
            'queue_avg_ms': timing_summary['queue_avg_ms'],
            'write_avg_ms': timing_summary['write_avg_ms'],
            'decode_calls': timing_summary['decode_calls'],
            'queue_calls': timing_summary['queue_calls'],
            'write_calls': timing_summary['write_calls'],
        }

        final_text = "".join(collected_chunks) if collected_chunks else result.texts[0]
        return metrics_dict, final_text


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    load_cfg = OVGenAI_LoadConfig(
        model_path="/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Llama-3.2-3B-Instruct-abliterated-OpenVINO/Llama-3.2-3B-Instruct-abliterated-int4_asym-ov",
        device="GPU.2"
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
        repetition_penalty=1.1,
        stream_chunk_tokens=3
    )

    text_gen = OVGenAI_Text2Text()
    text_gen.load_model(load_cfg)

    metrics, output = text_gen.generate_stream(textgeneration_config)
    print("\n\n=== Streaming Metrics ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== Output ===\n", output)
