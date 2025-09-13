import gc
import time
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass

import openvino as ov
from transformers import AutoTokenizer
from openvino_genai import (
    GenerationConfig,
    ContinuousBatchingPipeline,
    SchedulerConfig,
)

from pydantic import BaseModel, Field


@dataclass
class RequestInfo:
    """Simple request tracking without pipeline built-in IDs."""
    custom_id: str
    prompt: str
    created_at: float
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None


class ContinuousBatchConfig(BaseModel):
    max_num_batched_tokens: int = Field(default=2048)
    max_num_seqs: int = Field(default=48)
    cache_size: int = Field(default=6)
    dynamic_split_fuse: bool = Field(default=True)
    enable_prefix_caching: bool = Field(default=True)
    use_cache_eviction: bool = Field(default=False)


class OVGenAI_RequestManager:
    """
    Simple request manager without built-in pipeline request IDs.
    Uses pipeline.generate() instead of add_request().
    """

    def __init__(self, model_dir: str, device: str, batch_config: ContinuousBatchConfig):
        self.model_dir = model_dir
        self.device = device
        self.batch_config = batch_config
        self.pipeline = None
        self.tokenizer = None
        
        # Custom request tracking (no pipeline IDs)
        self.requests: Dict[str, RequestInfo] = {}
        self.request_counter = 0

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

    def submit_request(self, messages: List[Dict[str, str]], custom_id: Optional[str] = None) -> str:
        """
        Create a custom request (no pipeline IDs involved).
        Returns custom request ID for tracking.
        """
        if not custom_id:
            self.request_counter += 1
            custom_id = f"req_{self.request_counter}_{uuid.uuid4().hex[:8]}"
        
        # Convert messages to simple text for tracking
        prompt_text = " ".join([msg.get("content", "") for msg in messages])
        
        # Store request info
        request_info = RequestInfo(
            custom_id=custom_id,
            prompt=prompt_text,
            created_at=time.time()
        )
        self.requests[custom_id] = request_info
        
        return custom_id
    
    def process_request(self, custom_id: str, generation_config: GenerationConfig) -> str:
        """
        Process request using simple pipeline.generate() (no built-in request IDs).
        """
        if custom_id not in self.requests:
            return f"Error: Request {custom_id} not found"
        
        request_info = self.requests[custom_id]
        
        try:
            # Convert back to messages format for processing
            messages = [{"role": "user", "content": request_info.prompt}]
            encoded_prompt = self.prepare_inputs(messages)
            
            # Use simple generate method (no request IDs)
            results = self.pipeline.generate([encoded_prompt], [generation_config])
            
            if results and len(results) > 0:
                generated_text = results[0].texts[0] if results[0].texts else ""
                request_info.result = generated_text
                request_info.completed_at = time.time()
                return generated_text
            else:
                request_info.error = "No results generated"
                request_info.completed_at = time.time()
                return "Error: No results generated"
                
        except Exception as e:
            request_info.error = str(e)
            request_info.completed_at = time.time()
            return f"Error: {e}"
    
    def get_request_info(self, custom_id: str) -> Optional[RequestInfo]:
        """Get request information by custom ID."""
        return self.requests.get(custom_id)
    
    def list_requests(self) -> Dict[str, RequestInfo]:
        """List all tracked requests."""
        return self.requests.copy()


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

    print("=== Custom Request Tracking (No Pipeline IDs) ===")
    start = time.time()

    # Submit requests and process them immediately
    for i in range(N):
        # Submit request (creates custom ID, no pipeline involvement)
        custom_id = manager.submit_request(messages)
        print(f"\n[Request {i+1} | Custom ID: {custom_id}] ------------------------")
        
        # Process the request using pipeline.generate()
        output_text = manager.process_request(custom_id, generation_config)
        print(f"Result: {output_text}")
        
        # Show request info
        req_info = manager.get_request_info(custom_id)
        if req_info and req_info.completed_at:
            duration = req_info.completed_at - req_info.created_at
            print(f"Duration: {duration:.2f}s")

    end = time.time()
    print(f"\nTotal time for {N} requests: {end - start:.2f} s")
    
    # Show all tracked requests
    print("\n=== All Custom Tracked Requests ===")
    for custom_id, req_info in manager.list_requests().items():
        status = "✅ Completed" if req_info.result else "❌ Failed"
        print(f"{status} {custom_id}: {req_info.prompt[:50]}...")

    manager.unload_model()


if __name__ == "__main__":
    main()
