import logging
import asyncio
import uuid
import base64
import io
import torch
import soundfile as sf
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Tuple, Union

from src.engine.ov_genai.llm import OVGenAI_LLM
from src.engine.ov_genai.vlm import OVGenAI_VLM
from src.engine.ov_genai.whisper import OVGenAI_Whisper
from src.engine.openvino.kokoro import OV_Kokoro
from src.engine.optimum.optimum_emb import Optimum_EMB
from src.engine.optimum.optimum_rr import Optimum_RR

from src.server.models.openvino import OV_KokoroGenConfig
from src.server.models.ov_genai import OVGenAI_GenConfig, OVGenAI_WhisperGenConfig
from src.server.models.optimum import PreTrainedTokenizerConfig, RerankerConfig
from src.server.model_registry import ModelRecord, ModelRegistry
from src.server.models.registration import ModelType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    gen_config: Union[OVGenAI_GenConfig, OVGenAI_WhisperGenConfig, OV_KokoroGenConfig, PreTrainedTokenizerConfig]
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
    - infer_emb: Process embedding requests
    - infer_rerank: Process reranking requests
    """
    
    @staticmethod
    async def infer_llm(packet: WorkerPacket, llm_instance: OVGenAI_LLM, registry: 'WorkerRegistry' = None) -> WorkerPacket:
        """Generate text for a single packet using the OVGenAI_LLM pipeline"""
        metrics = None
        final_text = ""

        try:
            # Register model instance for cancellation tracking
            if registry is not None:
                async with registry._lock:
                    if packet.request_id in registry._active_requests:
                        model_name, _ = registry._active_requests[packet.request_id]
                        registry._active_requests[packet.request_id] = (model_name, llm_instance)

            async for item in llm_instance.arc_infer(packet.gen_config, packet.request_id):
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
        except Exception as e:
            # Log the full exception with traceback
            logger.error("LLM inference failed!", exc_info=True)
            # Store error in packet response
            packet.response = f"Error: {str(e)}"
            packet.metrics = None
            # Signal error to stream if streaming
            if packet.gen_config.stream and packet.stream_queue is not None:
                await packet.stream_queue.put(None)

        # Clean up active request tracking
        if registry is not None:
            async with registry._lock:
                registry._active_requests.pop(packet.request_id, None)

        return packet

    @staticmethod
    async def infer_vlm(packet: WorkerPacket, vlm_model: OVGenAI_VLM, registry: 'WorkerRegistry' = None) -> WorkerPacket:
        """Generate text from image for a single packet using the OVGenAI_VLM pipeline"""
        metrics = None
        final_text = ""

        try:
            # Register model instance for cancellation tracking
            if registry is not None:
                async with registry._lock:
                    if packet.request_id in registry._active_requests:
                        model_name, _ = registry._active_requests[packet.request_id]
                        registry._active_requests[packet.request_id] = (model_name, vlm_model)

            async for item in vlm_model.arc_infer(packet.gen_config, packet.request_id):
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
        except Exception as e:
            # Log the full exception with traceback
            logger.error("VLM inference failed!", exc_info=True)
            # Store error in packet response
            packet.response = f"Error: {str(e)}"
            packet.metrics = None
            # Signal error to stream if streaming
            if packet.gen_config.stream and packet.stream_queue is not None:
                await packet.stream_queue.put(None)

        # Clean up active request tracking
        if registry is not None:
            async with registry._lock:
                registry._active_requests.pop(packet.request_id, None)

        return packet

    @staticmethod
    async def infer_whisper(packet: WorkerPacket, whisper_model: OVGenAI_Whisper) -> WorkerPacket:
        """Transcribe audio for a single packet using the OVGenAI_Whisper pipeline.

        Note: Whisper pipeline operates non-streaming; this method processes the
        AsyncIterator to collect metrics and final text.
        """
        metrics = None
        final_text = ""

        try:
            async for item in whisper_model.transcribe(packet.gen_config):
                if isinstance(item, dict):
                    metrics = item
                else:
                    final_text = item

            packet.response = final_text
            packet.metrics = metrics
        except Exception as e:
            # Log the full exception with traceback
            logger.error("Whisper inference failed!", exc_info=True)
            # Store error in packet response
            packet.response = f"Error: {str(e)}"
            packet.metrics = None
            
        return packet

    @staticmethod
    async def infer_kokoro(packet: WorkerPacket, kokoro_model: OV_Kokoro) -> WorkerPacket:
        """Generate speech audio for a single packet using the OV_Kokoro pipeline.

        Collects audio chunks and concatenates them into a single audio tensor,
        then converts to bytes for response.
        """
        audio_chunks = []
        chunk_texts = []

        try:
            async for chunk in kokoro_model.chunk_forward_pass(packet.gen_config):
                audio_chunks.append(chunk.audio)
                chunk_texts.append(chunk.chunk_text)

            if audio_chunks:
                # Concatenate all audio chunks
                full_audio = torch.cat(audio_chunks, dim=0)

                # Convert to WAV bytes
                wav_buffer = io.BytesIO()
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
        except Exception as e:
            # Log the full exception with traceback
            logger.error("Kokoro inference failed!", exc_info=True)
            # Store error in packet response
            packet.response = f"Error: {str(e)}"
            packet.metrics = None

        return packet
    
    @staticmethod
    async def infer_emb(packet: WorkerPacket, emb_instance: Optimum_EMB) -> WorkerPacket:
        """Generate embeddings for a single packet using the optimum pipeline"""
        metrics = None
        final_data = None

        try:
            async for item in emb_instance.generate_embeddings(packet.gen_config):
                if isinstance(item, dict):
                    metrics = item
                else:
                    final_data = item

            packet.response = final_data
            packet.metrics = metrics
            
        except Exception as e:
            # Log the full exception with traceback
            logger.error("EMB inference failed!", exc_info=True)
            # Store error in packet response
            packet.response = f"Error: {str(e)}"
            packet.metrics = None
            # Signal error to stream if streaming
            if packet.gen_config.stream and packet.stream_queue is not None:
                await packet.stream_queue.put(None)
                
        return packet

    @staticmethod
    async def infer_rerank(packet: WorkerPacket, rerank_instance: Optimum_RR) -> WorkerPacket:
        """Generate reranking for a single packet using the optimum pipeline"""
        metrics = None
        final_data = None

        try:
            async for item in rerank_instance.generate_rerankings(packet.gen_config):
                if isinstance(item, dict):
                    metrics = item
                else:
                    final_data = item

            packet.response = final_data
            packet.metrics = metrics
            
        except Exception as e:
            # Log the full exception with traceback
            logger.error("Reranking failed!", exc_info=True)
            # Store error in packet response
            packet.response = f"Error: {str(e)}"
            packet.metrics = None
            # Signal error to stream if streaming
            if packet.gen_config.stream and packet.stream_queue is not None:
                await packet.stream_queue.put(None)
                
        return packet
    
class QueueWorker:
    """
    Manages inference worker loops for consuming and processing packets from model queues.

    Uses a factory pattern to create worker coroutines dynamically based on model type.
    """

    @staticmethod
    async def _generic_worker(
        model_name: str,
        model_queue: asyncio.Queue,
        model_instance: Any,
        registry: ModelRegistry,
        worker_type: str,
        infer_method: callable,
        error_check_fn: callable,
        worker_registry: 'WorkerRegistry' = None,
    ) -> None:
        """Generic worker loop that processes packets from queue using provided inference method."""
        logger.info(f"[{model_name} {worker_type} Worker] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                logger.info(f"[{model_name} {worker_type} Worker] Shutdown signal received.")
                break

            # Pass worker_registry for cancellation tracking
            completed_packet = await infer_method(packet, model_instance, worker_registry)

            # Check if inference failed and trigger model unload
            if error_check_fn(completed_packet):
                logger.error(f"[{model_name} {worker_type} Worker] Inference failed, triggering model unload...")
                asyncio.create_task(registry.register_unload(model_name))
                break

            if completed_packet.metrics:
                logger.info(f"[{model_name} {worker_type} Worker] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    def create_worker_queue(
        model_type: ModelType,
        model_name: str,
        model_queue: asyncio.Queue,
        model_instance: Any,
        registry: ModelRegistry,
        worker_registry: 'WorkerRegistry' = None,
    ) -> asyncio.Task:
        """Factory method to create the appropriate worker task based on model type."""
        # Error check functions
        def error_check_starts_with_error(packet: WorkerPacket) -> bool:
            return bool(packet.response and packet.response.startswith("Error:"))

        def error_check_falsy_response(packet: WorkerPacket) -> bool:
            return not packet.response

        # Worker configuration mapping
        worker_config = {
            ModelType.LLM: {
                "worker_type": "LLM",
                "infer_method": InferWorker.infer_llm,
                "error_check_fn": error_check_starts_with_error,
            },
            ModelType.VLM: {
                "worker_type": "VLM",
                "infer_method": InferWorker.infer_vlm,
                "error_check_fn": error_check_starts_with_error,
            },
            ModelType.WHISPER: {
                "worker_type": "Whisper",
                "infer_method": InferWorker.infer_whisper,
                "error_check_fn": error_check_starts_with_error,
            },
            ModelType.KOKORO: {
                "worker_type": "Kokoro",
                "infer_method": InferWorker.infer_kokoro,
                "error_check_fn": error_check_starts_with_error,
            },
            ModelType.EMB: {
                "worker_type": "EMB",
                "infer_method": InferWorker.infer_emb,
                "error_check_fn": error_check_falsy_response,
            },
            ModelType.RERANK: {
                "worker_type": "Reranker",
                "infer_method": InferWorker.infer_rerank,
                "error_check_fn": error_check_falsy_response,
            },
        }

        config = worker_config.get(model_type)
        if config is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        return asyncio.create_task(
            QueueWorker._generic_worker(
                model_name=model_name,
                model_queue=model_queue,
                model_instance=model_instance,
                registry=registry,
                worker_type=config["worker_type"],
                infer_method=config["infer_method"],
                error_check_fn=config["error_check_fn"],
                worker_registry=worker_registry,
            )
        )

class WorkerRegistry:
    """
    Central orchestrator for managing per-model inference workers and request routing.
    
    WorkerRegistry serves as the main coordination layer that bridges the ModelRegistry
    with the actual inference execution. It automatically spawns and manages dedicated
    worker tasks for each loaded model, routing generation requests to the appropriate
    model-specific queues.
    

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
        
        self._model_queues_emb: Dict[str, asyncio.Queue] = {}
        self._model_tasks_emb: Dict[str, asyncio.Task] = {}

        self._model_queues_rerank: Dict[str, asyncio.Queue] = {}
        self._model_tasks_rerank: Dict[str, asyncio.Task] = {}

        # Track active streaming requests for cancellation
        # request_id -> (model_name, model_instance)
        self._active_requests: Dict[str, Tuple[str, Any]] = {}

        self._lock = asyncio.Lock()

        self._model_registry.add_on_loaded(self._on_model_loaded)
        self._model_registry.add_on_unloaded(self._on_model_unloaded)

    def _normalize_model_type(self, mt) -> Optional[ModelType]:
        if isinstance(mt, ModelType):
            return mt
        try:
            return ModelType(mt)
        except Exception:
            return None

    async def _on_model_loaded(self, record: ModelRecord) -> None:
        mt = self._normalize_model_type(record.model_type)
        if mt is None:
            logger.info(f"[WorkerRegistry] Unknown model_type for {record.model_name}: {record.model_type}")
            return

        instance = record.model_instance

        async with self._lock:
            if mt == ModelType.LLM and isinstance(instance, OVGenAI_LLM):
                if record.model_name not in self._model_queues_llm:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_llm[record.model_name] = q
                    task = QueueWorker.create_worker_queue(mt, record.model_name, q, instance, self._model_registry, self)
                    self._model_tasks_llm[record.model_name] = task

            elif mt == ModelType.VLM and isinstance(instance, OVGenAI_VLM):
                if record.model_name not in self._model_queues_vlm:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_vlm[record.model_name] = q
                    task = QueueWorker.create_worker_queue(mt, record.model_name, q, instance, self._model_registry, self)
                    self._model_tasks_vlm[record.model_name] = task

            elif mt == ModelType.WHISPER and isinstance(instance, OVGenAI_Whisper):
                if record.model_name not in self._model_queues_whisper:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_whisper[record.model_name] = q
                    task = QueueWorker.create_worker_queue(mt, record.model_name, q, instance, self._model_registry)
                    self._model_tasks_whisper[record.model_name] = task

            elif mt == ModelType.KOKORO and isinstance(instance, OV_Kokoro):
                if record.model_name not in self._model_queues_kokoro:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_kokoro[record.model_name] = q
                    task = QueueWorker.create_worker_queue(mt, record.model_name, q, instance, self._model_registry)
                    self._model_tasks_kokoro[record.model_name] = task

            elif mt == ModelType.EMB and isinstance(instance, Optimum_EMB):
                if record.model_name not in self._model_queues_emb:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_emb[record.model_name] = q
                    task = QueueWorker.create_worker_queue(mt, record.model_name, q, instance, self._model_registry)
                    self._model_tasks_emb[record.model_name] = task

            elif mt == ModelType.RERANK and isinstance(instance, Optimum_RR):
                if record.model_name not in self._model_queues_rerank:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_rerank[record.model_name] = q
                    task = QueueWorker.create_worker_queue(mt, record.model_name, q, instance, self._model_registry)
                    self._model_tasks_rerank[record.model_name] = task
            else:
                logger.info(f"[WorkerRegistry] Model type/instance mismatch for {record.model_name}: {record.model_type}, {type(instance)}")

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

            # Try emb dicts
            q = self._model_queues_emb.pop(record.model_name, None)
            t = self._model_tasks_emb.pop(record.model_name, None)
            if q is not None:
                await q.put(None)
            if t is not None and not t.done():
                t.cancel()
                
            # Try rerank dicts
            q = self._model_queues_rerank.pop(record.model_name, None)
            t = self._model_tasks_rerank.pop(record.model_name, None)
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

    def _get_emb_queue(self, model_name: str) -> asyncio.Queue:
        q = self._model_queues_emb.get(model_name)
        if q is not None:
            return q
        raise ValueError(f"Embedding model '{model_name}' is not loaded or no worker is available")

    def _get_rerank_queue(self, model_name: str) -> asyncio.Queue:
        q = self._model_queues_rerank.get(model_name)
        if q is not None:
            return q
        raise ValueError(f"Rerank model '{model_name}' is not loaded or no worker is available")
    
    async def arc_generate(self, model_name: str, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Generate text using the arc_infer codepath, supporting both streaming and non-streaming.

        Unified entry point for LLM inference that delegates to llm.py arc_infer.
        Handles both streaming (stream=True) and non-streaming (stream=False) based on gen_config.stream.

        Args:
            model_name: Target model name
            gen_config: Generation configuration with stream flag

        Yields:
            For streaming (stream=True): Text chunks followed by metrics dict
            For non-streaming (stream=False): Single text chunk followed by metrics dict
        """
        request_id = uuid.uuid4().hex

        if gen_config.stream:
            # Streaming mode: use stream_queue for async iteration
            stream_queue: asyncio.Queue = asyncio.Queue()
            packet = WorkerPacket(
                request_id=request_id,
                id_model=model_name,
                gen_config=gen_config,
                stream_queue=stream_queue,
            )

            # Register active request for cancellation tracking
            async with self._lock:
                self._active_requests[request_id] = (model_name, None)

            q = self._get_model_queue(model_name)
            await q.put(packet)

            while True:
                item = await stream_queue.get()
                if item is None:
                    # Clean up active request tracking
                    async with self._lock:
                        self._active_requests.pop(request_id, None)
                    break
                yield item
        else:
            # Non-streaming mode: use result_future for single response
            result_future: asyncio.Future = asyncio.get_running_loop().create_future()
            packet = WorkerPacket(
                request_id=request_id,
                id_model=model_name,
                gen_config=gen_config,
                result_future=result_future,
            )

            # Register active request for cancellation tracking
            async with self._lock:
                self._active_requests[request_id] = (model_name, None)

            q = self._get_model_queue(model_name)
            await q.put(packet)
            completed = await result_future

            # Clean up active request tracking
            async with self._lock:
                self._active_requests.pop(request_id, None)

            # Yield the full response as a single chunk, then yield metrics
            yield completed.response or ""
            if completed.metrics:
                yield completed.metrics

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
    
    async def embed(self, model_name: str, tok_config: PreTrainedTokenizerConfig) -> Dict[str, Any]:
        """Create embeddings."""
        request_id = uuid.uuid4().hex
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        packet = WorkerPacket(
            request_id=request_id,
            id_model=model_name,
            gen_config=tok_config,
            result_future=result_future,
        )
        q = self._get_emb_queue(model_name)
        await q.put(packet)
        completed = await result_future
        return {"data": completed.response, "metrics": completed.metrics or {}}
    
    async def rerank(self, model_name: str, rr_config: RerankerConfig) -> Dict[str, Any]:
        """Rerank documents."""
        request_id = uuid.uuid4().hex
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        packet = WorkerPacket(
            request_id=request_id,
            id_model=model_name,
            gen_config=rr_config,
            result_future=result_future,
        )
        q = self._get_rerank_queue(model_name)
        await q.put(packet)
        completed = await result_future
        return {"data": completed.response, "metrics": completed.metrics or {}}

    async def cancel(self, request_id: str) -> bool:
        """
        Cancel an ongoing generation by request_id (works for both streaming and non-streaming).

        Args:
            request_id: The request ID to cancel

        Returns:
            True if cancellation was triggered, False if request_id not found
        """
        if request_id in self._active_requests:
            model_name, _ = self._active_requests[request_id]
            # Look up model instance from ModelRegistry
            async with self._model_registry._lock:
                for record in self._model_registry._models.values():
                    if record.model_name == model_name and record.model_instance is not None:
                        model_instance = record.model_instance
                        if hasattr(model_instance, 'cancel'):
                            await model_instance.cancel(request_id)
                            logger.info(f"[WorkerRegistry] Cancelled request {request_id} on model {model_name}")
                            return True
        return False