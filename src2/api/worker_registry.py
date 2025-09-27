import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any, AsyncIterator, Union

from src2.api.base_config import OVGenAI_GenConfig
from src2.api.model_registry import ModelRegistry, ModelRecord, TaskType
from src2.engine.ov_genai.ov_genai_llm import OVGenAI_LLM
from src2.engine.ov_genai.ov_genai_vlm import OVGenAI_VLM
from src2.engine.ov_genai.whisper import OVGenAI_Whisper, OVGenAI_WhisperGenConfig
from src2.engine.openvino.ov_kokoro import OV_Kokoro, OV_KokoroGenConfig


@dataclass
class WorkerPacket:
    """
    Data container for inference requests flowing through the worker system.

    WorkerPacket encapsulates all information needed to process a single generation
    request, including the request configuration, response data, and orchestration
    primitives for async communication between components.

    Request Flow:
    1. Created by WorkerRegistry with request_id, id_model, and gen_config
    2. Routed to appropriate model queue based on id_model
    3. Processed by Worker_ModelManager which delegates to Worker_QueueHandler
    4. Response and metrics populated during generation
    5. Results communicated back via result_future or stream_queue

    Fields:
    - request_id: Unique identifier for tracking and logging
    - id_model: Target model name for routing to correct worker
    - gen_config: Complete generation configuration (messages, parameters, etc.)
    - response: Final generated text (populated after processing)
    - metrics: Performance metrics from generation (tokens/sec, etc.)
    - result_future: Async communication for non-streaming requests
    - stream_queue: Async communication for streaming requests


    """
    request_id: str
    id_model: str  # model_name
    gen_config: Union[OVGenAI_GenConfig, OVGenAI_WhisperGenConfig, OV_KokoroGenConfig]
    response: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    # Orchestration plumbing
    result_future: Optional[asyncio.Future] = None
    stream_queue: Optional[asyncio.Queue] = None

class InferWorker:
    """
    Handles generation for individual packets.
    
    Responsibilities:
    - Execute generation requests using pipelines

    
    Methods:
    - infer_llm: Process text-to-text generation requests
    - infer_vlm: Process image-to-text generation requests
    - infer_whisper: Process audio transcription requests
    - infer_kokoro: Process speech generation requests

    
    Note: All methods are static as they operate on provided packets and generators
    without maintaining internal state.
    """
    
    @staticmethod
    async def infer_llm(packet: WorkerPacket, llm_instance: OVGenAI_LLM) -> WorkerPacket:
        """Generate text for a single packet using the OVGenAI_LLM pipeline"""
        metrics = None
        final_text = ""

        async for item in llm_instance.generate_type(packet.gen_config):
            if isinstance(item, dict):
                metrics = item
            else:
                if packet.gen_config.stream:
                    final_text += item
                    if packet.stream_queue is not None:
                        await packet.stream_queue.put(item)
                else:
                    final_text = item

        packet.response = final_text
        packet.metrics = metrics
        if packet.gen_config.stream and packet.stream_queue is not None:
            if metrics is not None:
                await packet.stream_queue.put({"metrics": metrics})
            await packet.stream_queue.put(None)
        return packet

    @staticmethod
    async def infer_vlm(packet: WorkerPacket, vlm_model: OVGenAI_VLM) -> WorkerPacket:
        """Generate text from image for a single packet using the OVGenAI_VLM pipeline"""
        metrics = None
        final_text = ""

        async for item in vlm_model.generate_type(packet.gen_config):
            if isinstance(item, dict):
                metrics = item
            else:
                if packet.gen_config.stream:
                    final_text += item
                    if packet.stream_queue is not None:
                        await packet.stream_queue.put(item)
                else:
                    final_text = item

        packet.response = final_text
        packet.metrics = metrics
        if packet.gen_config.stream and packet.stream_queue is not None:
            if metrics is not None:
                await packet.stream_queue.put({"metrics": metrics})
            await packet.stream_queue.put(None)
        return packet

    @staticmethod
    async def infer_whisper(packet: WorkerPacket, whisper_model: OVGenAI_Whisper) -> WorkerPacket:
        """Transcribe audio for a single packet using the OVGenAI_Whisper pipeline.

        Note: Whisper pipeline operates non-streaming; this method processes the
        AsyncIterator to collect metrics and final text.
        """
        metrics = None
        final_text = ""

        async for item in whisper_model.transcribe(packet.gen_config):
            if isinstance(item, dict):
                metrics = item
            else:
                final_text = item

        packet.response = final_text
        packet.metrics = metrics
        return packet

    @staticmethod
    async def infer_kokoro(packet: WorkerPacket, kokoro_model: OV_Kokoro) -> WorkerPacket:
        """Generate speech audio for a single packet using the OV_Kokoro pipeline.

        Collects audio chunks and concatenates them into a single audio tensor,
        then converts to bytes for response.
        """
        import torch
        import base64
        import io

        audio_chunks = []
        chunk_texts = []

        async for chunk in kokoro_model.chunk_forward_pass(packet.gen_config):
            audio_chunks.append(chunk.audio)
            chunk_texts.append(chunk.chunk_text)

        if audio_chunks:
            # Concatenate all audio chunks
            full_audio = torch.cat(audio_chunks, dim=0)

            # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            import soundfile as sf
            sf.write(wav_buffer, full_audio.numpy(), samplerate=24000, format='WAV')
            wav_bytes = wav_buffer.getvalue()

            # Encode as base64 for JSON response
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            packet.response = audio_base64
        else:
            packet.response = ""

        # Add some basic metrics
        packet.metrics = {
            "chunks_processed": len(audio_chunks),
            "chunk_texts": chunk_texts,
            "total_samples": sum(len(chunk) for chunk in audio_chunks) if audio_chunks else 0
        }

        return packet

class QueueWorker:
    """
    Manages inference worker loops for consuming and processing packets from model queues.
    
    This class orchestrates the continuous processing of inference requests by running
    dedicated worker loops for different model types. Each worker consumes packets from
    model-specific queues, delegates generation to Worker_QueueHandler, handles logging,
    and manages result communication back to callers.
    
    Responsibilities:
    - Run continuous worker loops for text and image models
    - Consume packets from model-specific asyncio queues
    - Coordinate with Worker_QueueHandler for actual generation
    - Extract and log user messages for different content types
    - Handle graceful shutdown via None packet signals
    - Manage result futures and task completion notifications
    
    Worker Types:
    - Text Workers (LLM): Handle text-to-text generation models
    - Image Workers (VLM): Handle image-to-text/vision-language models
    
    Methods:
    - worker_llm: Continuous worker for text models
    - worker_vlm: Continuous worker for image/multimodal models
    
    Architecture:
    Each worker runs as an independent asyncio task, processing packets sequentially
    while allowing multiple workers to operate in parallel for different models.
    Workers are spawned by WorkerRegistry and communicate via asyncio primitives.
    """
    
    @staticmethod
    async def queue_worker_llm(model_name: str, model_queue: asyncio.Queue, llm_model: OVGenAI_LLM):
        """Text model inference worker that processes packets from queue"""
        print(f"[{model_name} LLM Worker] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                print(f"[{model_name} LLM Worker] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_llm(packet, llm_model)

            if completed_packet.metrics:
                print(f"[{model_name} LLM Worker] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    async def queue_worker_vlm(model_name: str, model_queue: asyncio.Queue, vlm_model: OVGenAI_VLM):
        """Image model inference worker that processes packets from queue"""
        print(f"[{model_name} VLM Worker] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                print(f"[{model_name} VLM Worker] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_vlm(packet, vlm_model)

            if completed_packet.metrics:
                print(f"[{model_name} VLM Worker] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    async def queue_worker_whisper(model_name: str, model_queue: asyncio.Queue, whisper_model: OVGenAI_Whisper):
        """Whisper model inference worker that processes packets from queue"""
        print(f"[{model_name} Whisper Worker] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                print(f"[{model_name} Whisper Worker] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_whisper(packet, whisper_model)

            if completed_packet.metrics:
                print(f"[{model_name} Whisper Worker] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    async def queue_worker_kokoro(model_name: str, model_queue: asyncio.Queue, kokoro_model: OV_Kokoro):
        """Kokoro model inference worker that processes packets from queue"""
        print(f"[{model_name} Kokoro Worker] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                print(f"[{model_name} Kokoro Worker] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_kokoro(packet, kokoro_model)

            # Log the text that was converted to speech
            
            if completed_packet.metrics:
                print(f"[{model_name} Kokoro Worker] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

class WorkerRegistry:
    """
    Central orchestrator for managing per-model inference workers and request routing.
    
    WorkerRegistry serves as the main coordination layer that bridges the ModelRegistry
    with the actual inference execution. It automatically spawns and manages dedicated
    worker tasks for each loaded model, routing generation requests to the appropriate
    model-specific queues.
    
    Architecture Overview:
    - Subscribes to ModelRegistry events (model load/unload)
    - Maintains separate queue/task dictionaries per model type (text vs image)
    - Spawns Worker_ModelManager tasks for each loaded model
    - Provides high-level generate() and stream_generate() APIs
    - Handles request routing based on model names
    
    Key Features:
    - Automatic worker lifecycle management (spawn on load, cleanup on unload)
    - Type-safe model handling with explicit TaskType routing
    - Concurrent processing via per-model asyncio queues
    - Support for both streaming and non-streaming generation
    - Graceful shutdown and resource cleanup
    
    Data Structures:
    - _model_queues_llm/image: Per-model asyncio queues for request routing
    - _model_tasks_llm/image: Per-model asyncio tasks for worker management
    
    Public API:
    - generate(): Non-streaming text generation
    - stream_generate(): Streaming text generation with AsyncIterator
    
    Thread Safety:
    Uses asyncio.Lock for thread-safe access to internal dictionaries during
    concurrent model loading/unloading operations.
    """

    def __init__(self, model_registry: ModelRegistry):
        self._model_registry = model_registry

        # Separate queues/tasks per type for explicit control and future policies
        self._model_queues_llm: Dict[str, asyncio.Queue] = {}
        self._model_tasks_llm: Dict[str, asyncio.Task] = {}

        self._model_queues_vlm: Dict[str, asyncio.Queue] = {}
        self._model_tasks_vlm: Dict[str, asyncio.Task] = {}

        self._model_queues_whisper: Dict[str, asyncio.Queue] = {}
        self._model_tasks_whisper: Dict[str, asyncio.Task] = {}

        self._model_queues_kokoro: Dict[str, asyncio.Queue] = {}
        self._model_tasks_kokoro: Dict[str, asyncio.Task] = {}

        self._lock = asyncio.Lock()

        self._model_registry.add_on_loaded(self._on_model_loaded)
        self._model_registry.add_on_unloaded(self._on_model_unloaded)

    def _normalize_model_type(self, mt) -> Optional[TaskType]:
        if isinstance(mt, TaskType):
            return mt
        try:
            return TaskType(mt)
        except Exception:
            return None

    async def _on_model_loaded(self, record: ModelRecord) -> None:
        mt = self._normalize_model_type(record.model_type)
        if mt is None:
            print(f"[WorkerRegistry] Unknown model_type for {record.model_name}: {record.model_type}")
            return

        instance = record.model_instance

        async with self._lock:
            if mt == TaskType.TEXT_TO_TEXT and isinstance(instance, OVGenAI_LLM):
                if record.model_name not in self._model_queues_llm:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_llm[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_llm(record.model_name, q, instance))
                    self._model_tasks_llm[record.model_name] = task

            elif mt == TaskType.IMAGE_TO_TEXT and isinstance(instance, OVGenAI_VLM):
                if record.model_name not in self._model_queues_vlm:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_vlm[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_vlm(record.model_name, q, instance))
                    self._model_tasks_vlm[record.model_name] = task

            elif mt == TaskType.WHISPER and isinstance(instance, OVGenAI_Whisper):
                if record.model_name not in self._model_queues_whisper:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_whisper[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_whisper(record.model_name, q, instance))
                    self._model_tasks_whisper[record.model_name] = task

            elif mt == TaskType.KOKORO and isinstance(instance, OV_Kokoro):
                if record.model_name not in self._model_queues_kokoro:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_kokoro[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_kokoro(record.model_name, q, instance))
                    self._model_tasks_kokoro[record.model_name] = task

            else:
                print(f"[WorkerRegistry] Model type/instance mismatch for {record.model_name}: {record.model_type}, {type(instance)}")

    async def _on_model_unloaded(self, record: ModelRecord) -> None:
        async with self._lock:
            # Try text dicts
            q = self._model_queues_llm.pop(record.model_name, None)
            t = self._model_tasks_llm.pop(record.model_name, None)
            if q is not None:
                await q.put(None)
            if t is not None and not t.done():
                t.cancel()

            # Try image dicts
            q = self._model_queues_vlm.pop(record.model_name, None)
            t = self._model_tasks_vlm.pop(record.model_name, None)
            if q is not None:
                await q.put(None)
            if t is not None and not t.done():
                t.cancel()

            # Try whisper dicts
            q = self._model_queues_whisper.pop(record.model_name, None)
            t = self._model_tasks_whisper.pop(record.model_name, None)
            if q is not None:
                await q.put(None)
            if t is not None and not t.done():
                t.cancel()

            # Try kokoro dicts
            q = self._model_queues_kokoro.pop(record.model_name, None)
            t = self._model_tasks_kokoro.pop(record.model_name, None)
            if q is not None:
                await q.put(None)
            if t is not None and not t.done():
                t.cancel()

    def _get_model_queue(self, model_name: str) -> asyncio.Queue:
        q = self._model_queues_llm.get(model_name)
        if q is not None:
            return q
        q = self._model_queues_vlm.get(model_name)
        if q is not None:
            return q
        raise ValueError(f"Model '{model_name}' is not loaded or no worker is available")

    def _get_whisper_queue(self, model_name: str) -> asyncio.Queue:
        q = self._model_queues_whisper.get(model_name)
        if q is not None:
            return q
        raise ValueError(f"Whisper model '{model_name}' is not loaded or no worker is available")

    def _get_kokoro_queue(self, model_name: str) -> asyncio.Queue:
        q = self._model_queues_kokoro.get(model_name)
        if q is not None:
            return q
        raise ValueError(f"Kokoro model '{model_name}' is not loaded or no worker is available")

    async def generate(self, model_name: str, gen_config: OVGenAI_GenConfig) -> Dict[str, Any]:
        """Generate text without streaming."""
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

    async def stream_generate(self, model_name: str, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Generate text with streaming."""
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

    async def transcribe_whisper(self, model_name: str, gen_config: OVGenAI_WhisperGenConfig) -> Dict[str, Any]:
        """Transcribe audio using Whisper model."""

        request_id = uuid.uuid4().hex
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        packet = WorkerPacket(
            request_id=request_id,
            id_model=model_name,
            gen_config=gen_config,
            result_future=result_future,
        )
        q = self._get_whisper_queue(model_name)
        await q.put(packet)
        completed = await result_future
        return {"text": completed.response or "", "metrics": completed.metrics or {}}

    async def generate_speech_kokoro(self, model_name: str, gen_config: OV_KokoroGenConfig) -> Dict[str, Any]:
        """Generate speech using a loaded Kokoro model asynchronously via worker queue.

        Returns a dict with base64-encoded WAV audio and optional metrics.
        """
        request_id = uuid.uuid4().hex
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        packet = WorkerPacket(
            request_id=request_id,
            id_model=model_name,
            gen_config=gen_config,
            result_future=result_future,
        )
        q = self._get_kokoro_queue(model_name)
        await q.put(packet)
        completed = await result_future
        return {"audio_base64": completed.response or "", "metrics": completed.metrics or {}}