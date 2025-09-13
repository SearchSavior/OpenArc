import gc
import time
import uuid
from typing import List, Dict, Optional

import openvino as ov
from transformers import AutoTokenizer
from openvino_genai import (
    GenerationConfig,
    ContinuousBatchingPipeline,
    SchedulerConfig,
)

from pydantic import BaseModel, Field


class ContinuousBatchConfig(BaseModel):
    max_num_batched_tokens: int = Field(default=2048)
    max_num_seqs: int = Field(default=48)
    cache_size: int = Field(default=6)
    dynamic_split_fuse: bool = Field(default=True)
    enable_prefix_caching: bool = Field(default=True)
    use_cache_eviction: bool = Field(default=False)


class OVGenAI_RequestManager:
    """
    Dynamic request manager for ContinuousBatchingPipeline.
    Handles request submission, tracking, and result streaming.
    """

    def __init__(self, model_dir: str, device: str, batch_config: ContinuousBatchConfig):
        self.model_dir = model_dir
        self.device = device
        self.batch_config = batch_config
        self.pipeline = None
        self.tokenizer = None

    def load_model(self):
        """Initialize tokenizer and continuous batching pipeline."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            use_fast=True,
        )

        scheduler_config = SchedulerConfig()
        scheduler_config.max_num_batched_tokens = self.batch_config.max_num_batched_tokens
        scheduler_config.max_num_seqs = self.batch_config.max_num_seqs
        scheduler_config.cache_size = self.batch_config.cache_size
        scheduler_config.dynamic_split_fuse = self.batch_config.dynamic_split_fuse
        scheduler_config.enable_prefix_caching = self.batch_config.enable_prefix_caching
        scheduler_config.use_cache_eviction = self.batch_config.use_cache_eviction

        self.pipeline = ContinuousBatchingPipeline(
            self.model_dir,
            device=self.device,
            scheduler_config=scheduler_config,
        )

    def unload_model(self):
        """Clean up resources."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()

    def prepare_inputs(self, messages: List[Dict[str, str]]) -> ov.Tensor:
        """Convert chat messages → ov.Tensor of token IDs."""
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            skip_special_tokens=True,
            return_tensors="np",
        )
        return ov.Tensor(prompt_token_ids)

    def submit_prompt(self, messages: List[Dict[str, str]], generation_config: GenerationConfig, request_id: Optional[int] = None):
        """
        Submit a single prompt into the pipeline.
        Returns a request handle (GenerationHandle).
        """
        if not request_id:
            request_id = int(uuid.uuid4().int & 0x7FFFFFFF)  # Convert UUID to int within int32 range

        encoded_prompt = self.prepare_inputs(messages)
        request_handle = self.pipeline.add_request(
            request_id,
            encoded_prompt,
            generation_config,
        )
        return request_id, request_handle

    def stream_response(self, request_handle, decode=True):
        """
        Stream tokens until completion using GenerationHandle.
        """
        output_text = ""
        
        # Process the request until completion
        while request_handle.can_read():
            # Step the pipeline to generate more tokens
            self.pipeline.step()
            
            # Read available outputs
            if request_handle.can_read():
                generation_outputs = request_handle.read_all()
                for output in generation_outputs:
                    if decode:
                        # Decode the generated tokens
                        output_text += self.tokenizer.decode(output.generated_ids, skip_special_tokens=True)
                    else:
                        output_text += f"{output.generated_ids} "
        
        return output_text


def main():
    model_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen3-1.7B-int8_asym-ov"
    batch_config = ContinuousBatchConfig()

    manager = OVGenAI_RequestManager(model_dir=model_dir, device="GPU.1", batch_config=batch_config)
    manager.load_model()

    generation_config = GenerationConfig(
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    # Single prompt repeated N times
    N = 5
    messages = [{"role": "user", "content": "Tell me a quick fact about black holes."}]

    start = time.time()

    for i in range(N):
        req_id, req = manager.submit_prompt(messages, generation_config)
        print(f"\n[Request {i+1} | ID={req_id}] ------------------------")
        output_text = manager.stream_response(req)
        print(output_text)
        
        # Explicitly drop the request handle to help with cleanup
        req.drop()

    end = time.time()
    print(f"\nTotal time for {N} requests: {end - start:.2f} s")

    manager.unload_model()


if __name__ == "__main__":
    main()
