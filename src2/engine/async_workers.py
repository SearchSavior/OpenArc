import asyncio
from dataclasses import dataclass
from typing import Optional
from openvino_genai import LLMPipeline, GenerationConfig


@dataclass
class WorkerPacket:
    request_id: str
    prompt: str
    response: Optional[str] = None


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

async def generator_worker(packet: WorkerPacket, ov_pipeline: OV_Pipeline) -> WorkerPacket:
    """
    Generate text for a single packet using the OV pipeline.
    """
    result = await asyncio.to_thread(ov_pipeline.generate_text, packet.prompt)
    packet.response = result
    return packet


async def queue_worker(message_queue: asyncio.Queue, ov_pipeline: OV_Pipeline):
    """
    Consume packets from queue and delegate generation to generator_worker.
    """
    print("[Worker] Started, waiting for packets...")

    while True:
        packet = await message_queue.get()
        if packet is None:  # shutdown signal
            print("[Worker] Shutdown signal received.")
            break

        # Delegate to generator worker
        completed_packet = await generator_worker(packet, ov_pipeline)
        print(f"[Worker] Request {completed_packet.request_id}: {completed_packet.prompt!r} -> {completed_packet.response!r}")

        message_queue.task_done()


async def main():
    model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov"
    ov_pipeline = OV_Pipeline(model_path, device="GPU.2")

    message_queue = asyncio.Queue()

    # Start worker first
    worker_task = asyncio.create_task(queue_worker(message_queue, ov_pipeline))
    print("[Main] Worker started, model loaded into memory.")

    # Queue prompts as packets
    prompts = [
        "Hello world",
        "Explain quantum entanglement simply",
        "What is the capital of France?"
    ]
    for i, p in enumerate(prompts):
        packet = WorkerPacket(request_id=f"req_{i+1:03d}", prompt=p)
        await message_queue.put(packet)
        print(f"[Main] Queued packet: {packet.request_id} - {p!r}")

    # Wait for all packets to be processed
    await message_queue.join()

    # Clean shutdown
    await message_queue.put(None)
    await worker_task

    print("[Main] All packets processed, shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
