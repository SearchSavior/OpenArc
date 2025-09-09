import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, AsyncIterator, Union

from src2.api.base_config import OVGenAI_TextGenConfig
from src2.api.model_registry import ModelRegistry, ModelRecord
from src2.engine.ov_genai.text2text import OVGenAI_Text2Text


@dataclass
class WorkerPacket:
    request_id: str
    id_model: str  # New field to identify which model to use
    gen_config: OVGenAI_TextGenConfig  # Full generation configuration
    response: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    # Orchestration plumbing
    result_future: Optional[asyncio.Future] = None
    stream_queue: Optional[asyncio.Queue] = None


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
                if packet.stream_queue is not None:
                    await packet.stream_queue.put(item)
            else:
                # For non-streaming, this is the final text
                final_text = item
    
    packet.response = final_text
    packet.metrics = metrics
    if packet.gen_config.stream and packet.stream_queue is not None:
        if metrics is not None:
            await packet.stream_queue.put({"metrics": metrics})
        await packet.stream_queue.put(None)
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

        if packet.result_future is not None and not packet.result_future.done():
            packet.result_future.set_result(completed_packet)

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


class WorkerRegistry:
    """
    Orchestrates per-model inference workers and queues without using async_workers module.

    - Subscribes to ModelRegistry load/unload events
    - Starts an inference worker per model using local functions
    - Provides generate() and stream_generate() APIs
    """

    def __init__(self, model_registry: ModelRegistry):
        self._model_registry = model_registry
        self._model_queues: Dict[str, asyncio.Queue] = {}
        self._worker_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

        self._model_registry.add_on_loaded(self._on_model_loaded)
        self._model_registry.add_on_unloaded(self._on_model_unloaded)

    async def _on_model_loaded(self, record: ModelRecord) -> None:
        if not isinstance(record.model_instance, OVGenAI_Text2Text):
            return

        async with self._lock:
            model_name = record.model_name
            if model_name in self._model_queues:
                return
            model_queue: asyncio.Queue = asyncio.Queue()
            self._model_queues[model_name] = model_queue
            task = asyncio.create_task(inference_worker(model_name, model_queue, record.model_instance))
            self._worker_tasks[model_name] = task

    async def _on_model_unloaded(self, record: ModelRecord) -> None:
        async with self._lock:
            model_name = record.model_name
            model_queue = self._model_queues.get(model_name)
            task = self._worker_tasks.get(model_name)
            if model_queue is not None:
                await model_queue.put(None)
            if model_name in self._model_queues:
                del self._model_queues[model_name]
            if model_name in self._worker_tasks:
                del self._worker_tasks[model_name]
            if task is not None and not task.done():
                task.cancel()

    def _get_model_queue(self, model_name: str) -> asyncio.Queue:
        q = self._model_queues.get(model_name)
        if q is None:
            raise ValueError(f"Model '{model_name}' is not loaded or no worker is available")
        return q

    async def generate(self, model_name: str, gen_config: OVGenAI_TextGenConfig) -> Dict[str, Any]:
        if gen_config.stream:
            raise ValueError("Use stream_generate for streaming requests")
        request_id = uuid.uuid4().hex
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        packet = WorkerPacket(
            request_id=request_id,
            id_model=model_name,
            gen_config=gen_config,
            result_future=result_future,
        )
        q = self._get_model_queue(model_name)
        await q.put(packet)
        completed = await result_future
        return {"text": completed.response or "", "metrics": completed.metrics or {}}

    async def stream_generate(self, model_name: str, gen_config: OVGenAI_TextGenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        if not gen_config.stream:
            raise ValueError("Use generate for non-streaming requests")
        request_id = uuid.uuid4().hex
        stream_queue: asyncio.Queue = asyncio.Queue()
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        packet = WorkerPacket(
            request_id=request_id,
            id_model=model_name,
            gen_config=gen_config,
            stream_queue=stream_queue,
            result_future=result_future,
        )
        q = self._get_model_queue(model_name)
        await q.put(packet)
        while True:
            item = await stream_queue.get()
            if item is None:
                break
            yield item