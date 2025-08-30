import asyncio
from dataclasses import dataclass
from typing import Optional, Dict
from openvino_genai import LLMPipeline, GenerationConfig


@dataclass
class WorkerPacket:
    request_id: str
    id_model: str  # New field to identify which model to use
    prompt: str
    response: Optional[str] = None


class OV_Pipeline:
    """Thin wrapper around OpenVINO LLMPipeline for CPU inference."""

    def __init__(self, model_path: str, device: str = "CPU", id_model: str = "default"):
        # Initialize synchronous pipeline
        self.pipeline = LLMPipeline(model_path, device=device)
        self.id_model = id_model
        self.gen_config = GenerationConfig(max_new_tokens=128)

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


async def inference_worker(id_model: str, model_queue: asyncio.Queue, ov_pipeline: OV_Pipeline):
    """
    Consume packets from a model-specific queue and delegate generation.
    """
    print(f"[{id_model} Worker] Started, waiting for packets...")

    while True:
        packet = await model_queue.get()
        if packet is None:  # shutdown signal
            print(f"[{id_model} Worker] Shutdown signal received.")
            break

        # Delegate to generator worker
        completed_packet = await generator_worker(packet, ov_pipeline)
        print(f"[{id_model} Worker] Request {completed_packet.request_id}: {completed_packet.prompt!r} -> {completed_packet.response!r}")

        model_queue.task_done()


async def packet_router(input_queue: asyncio.Queue, model_queues: Dict[str, asyncio.Queue]):
    """
    Route incoming packets to the appropriate model queue based on id_model.
    """
    print("[Router] Started, routing packets to model queues...")
    
    while True:
        packet = await input_queue.get()
        if packet is None:  # shutdown signal
            print("[Router] Shutdown signal received.")
            # Forward shutdown signal to all model queues
            for model_queue in model_queues.values():
                await model_queue.put(None)
            break

        # Route to appropriate model queue
        if packet.id_model in model_queues:
            await model_queues[packet.id_model].put(packet)
            print(f"[Router] Routed {packet.request_id} to {packet.id_model} queue")
        else:
            print(f"[Router] ERROR: Unknown model {packet.id_model} for request {packet.request_id}")

        input_queue.task_done()


async def main():
    # Configure multiple models (hardcoded for now)
    model_configs = {
        "llama3": {
            "path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Phi/phi-4-int4_asym-awq-ov",
            "device": "GPU.2"
        },
        "llama70b": {
            "path": "/mnt/Ironwolf-4TB/Models/OpenVINO/Phi/phi-4-int4_asym-awq-ov", 
            "device": "GPU.1"
        }
    }

    # Create pipelines and queues for each model
    pipelines = {}
    model_queues = {}
    worker_tasks = []

    for id_model, config in model_configs.items():
        # Create pipeline
        pipeline = OV_Pipeline(config["path"], config["device"], id_model)
        pipelines[id_model] = pipeline
        
        # Create model-specific queue
        model_queue = asyncio.Queue()
        model_queues[id_model] = model_queue
        
        # Start worker for this model
        worker_task = asyncio.create_task(
            inference_worker(id_model, model_queue, pipeline)
        )
        worker_tasks.append(worker_task)
        print(f"[Main] {id_model} worker started")

    # Create main input queue and router
    input_queue = asyncio.Queue()
    router_task = asyncio.create_task(packet_router(input_queue, model_queues))
    print("[Main] Router started")

    # Queue test packets for both models
    test_packets = [
        WorkerPacket(request_id="req_001", id_model="llama3", prompt="Hello world"),
        WorkerPacket(request_id="req_002", id_model="llama70b", prompt="Explain quantum entanglement simply"),
        WorkerPacket(request_id="req_003", id_model="llama3", prompt="What is the capital of France?"),
        WorkerPacket(request_id="req_004", id_model="llama3", prompt="What is the capital of France?"),
        WorkerPacket(request_id="req_005", id_model="llama3", prompt="You suck doney balls"),
        WorkerPacket(request_id="req_006", id_model="llama3", prompt="Respect my authoritah"),
        WorkerPacket(request_id="req_007", id_model="llama3", prompt="What is the capital of France?"),
        WorkerPacket(request_id="req_008", id_model="llama3", prompt="What is the capital of France?"),
        WorkerPacket(request_id="req_009", id_model="llama3", prompt="What is the capital of France?"),
        WorkerPacket(request_id="req_010", id_model="llama3", prompt="What is the capital of France?"),
        WorkerPacket(request_id="req_011", id_model="llama70b", prompt="Write a haiku about coding"),
    ]
    
    for packet in test_packets:
        await input_queue.put(packet)
        print(f"[Main] Queued packet: {packet.request_id} for {packet.id_model}")

    # Wait for all packets to be processed
    await input_queue.join()
    for model_queue in model_queues.values():
        await model_queue.join()

    # Clean shutdown
    await input_queue.put(None)  # Signal router to shutdown
    await router_task
    
    # Wait for all workers to finish
    for worker_task in worker_tasks:
        await worker_task

    print("[Main] All packets processed, shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
