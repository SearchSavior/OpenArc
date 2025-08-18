import json
import queue
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Union

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
        default=512,
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
    
    stream: bool = Field(
        default=False,
        description="Stream output in chunks of tokens."
    )

    stream_chunk_tokens: int = Field(
        default=1,
        description="Stream chunk size in tokens. Must be greater than 0. If set > 1, stream output in chunks of this many tokens using ChunkStreamer."
    )


class ChunkStreamer(StreamerBase):
    """
    Streams decoded text in chunks of N tokens.
    - tokens_len == 1 → token-by-token streaming.
    - tokens_len  > 1 → emit after every N tokens.
    Uses cumulative decode + delta slicing to avoid subword boundary artifacts.
    """
    def __init__(self, tokenizer, config: OVGenAI_TextGenConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_len = (config.stream_chunk_tokens)  # enforce at least 1
        self.tokens_cache: List[int] = []          # cumulative token buffer
        self.since_last_emit: int = 0              # tokens collected since last emit
        self.last_print_len: int = 0               # length of decoded text we've already emitted
        self.text_queue: "queue.Queue[Optional[str]]" = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration
        return value

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        # Normalize input to a list of ints
        if isinstance(token, list):
            self.tokens_cache.extend(token)
            self.since_last_emit += len(token)
        else:
            self.tokens_cache.append(token)
            self.since_last_emit += 1

        # Only emit when we've reached the chunk boundary
        if self.since_last_emit >= self.tokens_len:
            text = self.tokenizer.decode(self.tokens_cache)
            # Emit only the newly materialized portion
            if len(text) > self.last_print_len:
                chunk = text[self.last_print_len:]
                if chunk:
                    self.text_queue.put(chunk)
                self.last_print_len = len(text)
            self.since_last_emit = 0

        return openvino_genai.StreamingStatus.RUNNING

    def end(self) -> None:
        # Flush any remaining tokens at the end
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.last_print_len:
            chunk = text[self.last_print_len:]
            if chunk:
                self.text_queue.put(chunk)
        # Signal completion
        self.text_queue.put(None)


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
        # If streaming is requested, delegate to the streaming method
        if config.stream:
            return self.generate_stream(config)
        

        generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )

        result = self.model.generate([config.conversation], generation_config)
        perf_metrics = result.perf_metrics

        metrics_dict = {
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
            'generate_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2),
            'input_tokens': len(config.conversation),
            'new_tokens': len(result.texts[0]),
            'total_tokens': len(config.conversation) + len(result.texts[0]),
        }
        return metrics_dict, result.texts[0]

    def generate_stream(self, config: OVGenAI_TextGenConfig):
        """
        Streaming text generation with profiling.
        """
        # If streaming is not requested, fall back to non-streaming generation
        if not config.stream:
            return self.generate_stream(config)

        generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty
        )

        tokenizer = self.model.get_tokenizer()
        streamer = ChunkStreamer(tokenizer, config)
        
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

        metrics_dict = {
            'num_generated_tokens': perf_metrics.get_num_generated_tokens(),
            'stream' : config.stream,
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 2),
            'generate_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 2),
            'input_tokens': perf_metrics.get_num_input_tokens(),
            'new_tokens': perf_metrics.get_num_generated_tokens(),
            'total_tokens': perf_metrics.get_num_input_tokens() + perf_metrics.get_num_generated_tokens(),
        }

        final_text = "".join(collected_chunks) if collected_chunks else result.texts[0]
        return metrics_dict, final_text


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    load_cfg = OVGenAI_LoadConfig(
        model_path="/mnt/Ironwolf-4TB/Models/OpenVINO/Phi/Phi-lthy4-OpenVINO/Phi-lthy4-int4_sym-awq-ov",
        device="GPU.1"
    )

    conversation_messages = [
        {"role": "system", "content": "Alway's talk like you are Pete, the succint, punctual and self-deprecating pirate captain."},
        {"role": "user", "content": "Man it stinks in here"},
        [{"role": "assistant", "content": "Arrr matey! The stench be foul, but we'll be smelling the sea air soon enough."}],
        [{"role": "user", "content": "You bet. Hey, thanks for the lift. What'd you say your name was?"}]
    ]
    conversation_json = json.dumps(conversation_messages, indent=2)

    textgeneration_config = OVGenAI_TextGenConfig(
        conversation=conversation_json,
        max_new_tokens=128,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
        stream_chunk_tokens=3,
        stream=True
    )

    text_gen = OVGenAI_Text2Text()
    text_gen.load_model(load_cfg)

    metrics, output = text_gen.generate_text(textgeneration_config)
    print("\n\nOutput")
    print("-"*100)
    print(output)
    print("\n\n")
    print("\n\nPerformance Metrics")
    print("-"*100)
    print(json.dumps(metrics, indent=2))
