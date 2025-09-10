import gc
import asyncio
import json
from typing import Any, Dict, List, Union, AsyncIterator

import logging
from transformers import AutoTokenizer
import openvino as ov
from openvino_genai import (
    GenerationConfig, 
    LLMPipeline,
    )

from src2.api.base_config import OVGenAI_TextGenConfig

from src2.api.model_registry import ModelLoadConfig, ModelRegistry, EngineType, ModelType
from src2.engine.ov_genai.streamers import ChunkStreamer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OVGenAI_Text2Text:
    def __init__(self, load_config: ModelLoadConfig):
        self.model_path = None
        self.encoder_tokenizer = None
        self.load_config = load_config

    def prepare_inputs(self, messages: List[Dict[str, str]]) -> ov.Tensor:
        """
        Convert a messages (list of {role, content}) into ov.Tensor using the cached AutoTokenizer
        and its chat template.

        apply_chat_template can be configured to return a numpy array, 
        which we then convert to an ov.Tensor the runtime can accept

        Args:
            messages: List[Dict[str, str]]

        returns:
            prompt_token_ids: 
        """
        prompt_token_ids = self.encoder_tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            skip_special_tokens=True,
            return_tensors="np"
            )
        return ov.Tensor(prompt_token_ids)
    
    def generate_type(self, gen_config: OVGenAI_TextGenConfig):
        """
        Unified text generation method that routes to streaming or non-streaming
        based on the stream flag in gen_config. Both paths return an async iterator.
        
        Args:
            gen_config: Configuration containing the stream flag and other parameters
            
        Returns:
            - Non-streaming: async iterator yielding [metrics: dict, new_text: str]
            - Streaming: async iterator yielding token chunks (str)... then [metrics: dict, new_text: str]
        """
        if gen_config.stream:
            return self.generate_stream(gen_config)
        else:
            return self.generate_text(gen_config)
    
    async def generate_text(self, gen_config: OVGenAI_TextGenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """
        Async non-streaming text generation.
        Yields in order: metrics (dict), new_text (str).
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

        prompt_token_ids = self.prepare_inputs(gen_config.messages)
        result = await asyncio.to_thread(self.model_path.generate, prompt_token_ids, generation_kwargs)
        
        perf_metrics = result.perf_metrics
        decoder_tokenizer = self.model_path.get_tokenizer()
        text = decoder_tokenizer.decode(result.tokens)[0] if getattr(result, "tokens", None) else ""

        metrics_dict = self.collect_metrics(gen_config, perf_metrics)
        yield metrics_dict
        yield text

    async def generate_stream(self, gen_config: OVGenAI_TextGenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Async streaming text generation.
        Yields token chunks (str) as they arrive, then metrics (dict), then final new_text (str).
        """
        generation_kwargs = GenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty
        )

        decoder_tokenizer = self.model_path.get_tokenizer()
        streamer = ChunkStreamer(decoder_tokenizer, gen_config)
        prompt_token_ids = self.prepare_inputs(gen_config.messages)

        async def _run_generation():
            return await asyncio.to_thread(
                self.model_path.generate,
                prompt_token_ids,
                generation_kwargs,
                streamer
            )

        gen_task = asyncio.create_task(_run_generation())

        try:
            while True:
                chunk = await streamer.text_queue.get()
                if chunk is None:
                    break
                yield chunk

        finally:
            result = await gen_task
            perf_metrics = result.perf_metrics
            metrics = self.collect_metrics(gen_config, perf_metrics)
            
            yield metrics
    
    def collect_metrics(self, gen_config: OVGenAI_TextGenConfig, perf_metrics) -> Dict[str, Any]:
        """
        Collect and format performance metrics into a dictionary.
        
        Args:
            gen_config: OVGenAI_TextGenConfig
            perf_metrics: PerfMetrics

        Returns:
            metrics: Dict[str, Any]
            """
        # Compute prefill throughput = input tokens / ttft (in seconds)
        ttft_seconds = perf_metrics.get_ttft().mean / 1000
        input_tokens = perf_metrics.get_num_input_tokens()
        prefill_throughput = round(input_tokens / ttft_seconds, 2) if ttft_seconds > 0 else 0

        metrics: Dict[str, Any] = {
            'load_time (s)': round(perf_metrics.get_load_time() / 1000, 2),
            'ttft (s)': round(perf_metrics.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics.get_tpot().mean, 5),
            'prefill_throughput (tokens/s)': prefill_throughput,
            'decode_throughput (tokens/s)': round(perf_metrics.get_throughput().mean, 5),
            'decode_duration (s)': round(perf_metrics.get_generate_duration().mean / 1000, 5),
            'input_token': input_tokens,
            'new_token': perf_metrics.get_num_generated_tokens(),
            'total_token': input_tokens + perf_metrics.get_num_generated_tokens(),
            'stream': gen_config.stream,
        }
        # Include streaming-specific fields
        if gen_config.stream and hasattr(gen_config, "stream_chunk_tokens"):
            metrics['stream_chunk_tokens'] = gen_config.stream_chunk_tokens
        
        return metrics

    def load_model(self, loader: ModelLoadConfig):
        """Load model using a ModelLoadConfig configuration and cache the tokenizer.

        Args:
            loader: ModelLoadConfig containing model_path, device, engine, and runtime_config.
        """
        if loader.engine != EngineType.OV_GENAI:
            raise ValueError(
                f"Engine '{loader.engine}' is not supported by OVGenAI_Text2Text. Use '{EngineType.OV_GENAI}'."
            )

        self.model_path = LLMPipeline(
            loader.model_path,
            loader.device,
            **(loader.runtime_config or {})
        )

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(loader.model_path)
        logging.info(f"Model loaded successfully: {loader.model_name}")

    async def unload_model(self, registry: ModelRegistry, model_id: str) -> bool:
        """Unregister model from registry and free memory resources.

        Args:
            registry: ModelRegistry to unregister from
            model_id: Private model identifier returned by register_load

        Returns:
            True if the model was found and unregistered, else False.
        """
        removed = await registry.register_unload(model_id)

        if hasattr(self, 'model_path') and self.model_path is not None:
            del self.model_path
            self.model_path = None
        
        if hasattr(self, 'encoder_tokenizer') and self.encoder_tokenizer is not None:
            del self.encoder_tokenizer
            self.encoder_tokenizer = None
        
        gc.collect()
        logging.info(f"[{self.load_config.model_name}] unloaded and memory cleaned up")
        return removed


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    async def _demo():
        loader = ModelLoadConfig(
            model_path="/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Llama-3.2-3B-Instruct-abliterated-OpenVINO/Llama-3.2-3B-Instruct-abliterated-int4_asym-ov",
            model_name="Llama-3.2-3B-Instruct-ov-int4",
            model_type=ModelType.TEXT_TO_TEXT,
            engine=EngineType.OV_GENAI,
            device="GPU.2",
            runtime_config={}
        )

        messages = [
            {"role": "system", "content": "Alway's talk like you are Pete, the succint, punctual and self-deprecating pirate captain."},
            {"role": "user", "content": "Man it stinks in here"},
            {"role": "assistant", "content": "Arrr matey! The stench be foul, but we'll be smelling the sea air soon enough."},
            {"role": "user", "content": "You bet. Hey, thanks for the lift. What'd you say your name was?"}
        ]

        textgeneration_gen_config = OVGenAI_TextGenConfig(
            messages=messages,
            max_new_tokens=128,
            temperature=0.5,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            stream_chunk_tokens=1,
            stream=True
        )

        text_gen = OVGenAI_Text2Text(ModelLoadConfig(model_path=loader.model_path, device=loader.device))
        text_gen.load_model(loader)

        received_metrics = False
        metrics = None
        final_text = None
        async for item in text_gen.generate_stream(textgeneration_gen_config):
            if isinstance(item, dict):
                metrics = item
                received_metrics = True
            else:
                if received_metrics:
                    final_text = item  # final consolidated text
                else:
                    logging.info(item)  # stream chunks as they come

        if metrics is not None:
            logging.info("\n\nPerformance Metrics")
            logging.info("-"*20)
            logging.info(json.dumps(metrics, indent=2))
        if final_text is not None:
            pass  # available if needed: final_text


    asyncio.run(_demo())
