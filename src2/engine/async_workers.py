import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any
from src2.engine.text2text import OVGenAI_Text2Text
from src2.api.base_config import OVGenAI_LoadConfig, OVGenAI_TextGenConfig





@dataclass
class WorkerPacket:
    request_id: str
    id_model: str  # New field to identify which model to use
    gen_config: OVGenAI_TextGenConfig  # Full generation configuration
    response: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


# ----------------- Async Orchestration -----------------

async def generator_worker(packet: WorkerPacket, text_generator: OVGenAI_Text2Text) -> WorkerPacket:
    """
    Generate text for a single packet using the OVGenAI_Text2Text pipeline.
    Supports both streaming and non-streaming generation.
    """
    metrics = None
    final_text = ""
    
    # Use the generate_type method which routes to streaming or non-streaming
    async for item in text_generator.generate_type(packet.gen_config):
        if isinstance(item, dict):
            # This is metrics
            metrics = item
        else:
            # This is text (either final text for non-streaming or chunk for streaming)
            if packet.gen_config.stream:
                # For streaming, we accumulate chunks into final text
                final_text += item
            else:
                # For non-streaming, this is the final text
                final_text = item
    
    packet.response = final_text
    packet.metrics = metrics
    return packet


async def inference_worker(id_model: str, model_queue: asyncio.Queue, text_generator: OVGenAI_Text2Text):
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
        completed_packet = await generator_worker(packet, text_generator)
        
        # Extract prompt from messages for logging
        user_message = next((msg["content"] for msg in packet.gen_config.messages if msg["role"] == "user"), "")
        print(f"[{id_model} Worker] Request {completed_packet.request_id}: {user_message!r} -> {completed_packet.response!r}")
        if completed_packet.metrics:
            print(f"[{id_model} Worker] Metrics: {completed_packet.metrics}")

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

    # Create text generators and queues for each model
    text_generators = {}
    model_queues = {}
    worker_tasks = []

    for id_model, config in model_configs.items():
        # Create load config and text generator
        load_config = OVGenAI_LoadConfig(
            id_model=config["path"],
            device=config["device"]
        )
        text_generator = OVGenAI_Text2Text(load_config)
        text_generator.load_model()
        text_generators[id_model] = text_generator
        
        # Create model-specific queue
        model_queue = asyncio.Queue()
        model_queues[id_model] = model_queue
        
        # Start worker for this model
        worker_task = asyncio.create_task(
            inference_worker(id_model, model_queue, text_generator)
        )
        worker_tasks.append(worker_task)
        print(f"[Main] {id_model} worker started")

    # Create main input queue and router
    input_queue = asyncio.Queue()
    router_task = asyncio.create_task(packet_router(input_queue, model_queues))
    print("[Main] Router started")

    # Queue test packets for both models with proper message format
    test_packets = [
        WorkerPacket(
            request_id="req_001",
            id_model="llama3",
            gen_config=OVGenAI_TextGenConfig(
                messages=[{"role": "user", "content": "Hello world"}],
                max_new_tokens=64,
                temperature=0.7,
                stream=False
            )
        ),
        WorkerPacket(
            request_id="req_002",
            id_model="llama70b",
            gen_config=OVGenAI_TextGenConfig(
                messages=[{"role": "user", "content": "Explain quantum entanglement simply"}],
                max_new_tokens=128,
                temperature=0.5,
                stream=True  # Test streaming
            )
        ),
        WorkerPacket(
            request_id="req_003",
            id_model="llama3",
            gen_config=OVGenAI_TextGenConfig(
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                max_new_tokens=32,
                temperature=0.3,
                stream=False
            )
        ),
        WorkerPacket(
            request_id="req_004",
            id_model="llama3",
            gen_config=OVGenAI_TextGenConfig(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Tell me a short joke"}
                ],
                max_new_tokens=64,
                temperature=0.8,
                stream=True  # Test streaming
            )
        ),
        WorkerPacket(
            request_id="req_005",
            id_model="llama3",
            gen_config=OVGenAI_TextGenConfig(
                messages=[{"role": "user", "content": "Write a one-line summary of machine learning"}],
                max_new_tokens=48,
                temperature=0.4,
                stream=False
            )
        ),
        WorkerPacket(
            request_id="req_006",
            id_model="llama70b",
            gen_config=OVGenAI_TextGenConfig(
                messages=[{"role": "user", "content": "Write a haiku about coding"}],
                max_new_tokens=64,
                temperature=0.6,
                stream=True  # Test streaming
            )
        ),
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
