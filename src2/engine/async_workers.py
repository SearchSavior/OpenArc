import asyncio
from openvino_genai import LLMPipeline, GenerationConfig


class OV_Pipeline:
    """Thin wrapper around OpenVINO LLMPipeline for CPU inference."""

    def __init__(self, model_path: str, device: str = "CPU"):
        # Initialize synchronous pipeline
        self.pipeline = LLMPipeline(model_path, device=device)
        # Default generation config (can be tuned)
        self.gen_config = GenerationConfig(max_new_tokens=32)

    def generate_text(self, prompt: str) -> str:
        chunks = []
        for token in self.pipeline.generate(prompt, self.gen_config):
            chunks.append(token)
        return "".join(chunks)



# ----------------- Async Orchestration -----------------

async def generator_worker(prompt: str, ov_pipeline: OV_Pipeline) -> str:
    """
    Generate text for a single prompt using the OV pipeline.
    """
    result = await asyncio.to_thread(ov_pipeline.generate_text, prompt)
    return result


async def queue_worker(message_queue: asyncio.Queue, ov_pipeline: OV_Pipeline):
    """
    Consume prompts from queue and delegate generation to generator_worker.
    """
    print("[Worker] Started, waiting for prompts...")

    while True:
        prompt = await message_queue.get()
        if prompt is None:  # shutdown signal
            print("[Worker] Shutdown signal received.")
            break

        # Delegate to generator worker
        result = await generator_worker(prompt, ov_pipeline)
        print(f"[Worker] Prompt: {prompt!r} -> {result!r}")

        message_queue.task_done()


async def main():
    model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov"
    ov_pipeline = OV_Pipeline(model_path, device="GPU.2")

    message_queue = asyncio.Queue()

    # Start worker first
    worker_task = asyncio.create_task(queue_worker(message_queue, ov_pipeline))
    print("[Main] Worker started, model loaded into memory.")

    # Queue prompts
    prompts = [
        "Hello world",
        "Explain quantum entanglement simply",
        "What is the capital of France?"
    ]
    for p in prompts:
        await message_queue.put(p)
        print(f"[Main] Queued prompt: {p!r}")

    # Wait for all prompts to be processed
    await message_queue.join()

    # Clean shutdown
    await message_queue.put(None)
    await worker_task

    print("[Main] All prompts processed, shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
