import logging
import asyncio
import uuid
import base64
import io
import torch
import soundfile as sf
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Union

from src.engine.ov_genai.llm import OVGenAI_LLM
from src.engine.ov_genai.vlm import OVGenAI_VLM
from src.engine.ov_genai.whisper import OVGenAI_Whisper
from src.engine.openvino.kokoro import OV_Kokoro
from src.engine.openvino.qwen3_asr.infer import OVQwen3ASR, Qwen3ASRConfig
from src.engine.optimum.optimum_emb import Optimum_EMB
from src.engine.optimum.optimum_rr import Optimum_RR

from src.server.models.openvino import OV_KokoroGenConfig, OV_Qwen3ASRGenConfig
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
    gen_config: Union[
        OVGenAI_GenConfig,
        OVGenAI_WhisperGenConfig,
        OV_Qwen3ASRGenConfig,
        OV_KokoroGenConfig,
        PreTrainedTokenizerConfig,
    ]
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
    async def infer_llm(packet: WorkerPacket, llm_instance: OVGenAI_LLM) -> WorkerPacket:
        """Generate text for a single packet using the OVGenAI_LLM pipeline"""
        metrics = None
        final_text = ""

        try:
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
        except Exception as e:
            # Log the full exception with traceback
            logger.error("LLM inference failed!", exc_info=True)
            # Store error in packet response
            packet.response = f"Error: {str(e)}"
            packet.metrics = None
            # Signal error to stream if streaming
            if packet.gen_config.stream and packet.stream_queue is not None:
                await packet.stream_queue.put(None)
                
        return packet

    @staticmethod
    async def infer_vlm(packet: WorkerPacket, vlm_model: OVGenAI_VLM) -> WorkerPacket:
        """Generate text from image for a single packet using the OVGenAI_VLM pipeline"""
        metrics = None
        final_text = ""

        try:
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
        except Exception as e:
            # Log the full exception with traceback
            logger.error("VLM inference failed!", exc_info=True)
            # Store error in packet response
            packet.response = f"Error: {str(e)}"
            packet.metrics = None
            # Signal error to stream if streaming
            if packet.gen_config.stream and packet.stream_queue is not None:
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
    async def infer_qwen3_asr(packet: WorkerPacket, asr_model: OVQwen3ASR) -> WorkerPacket:
        """Transcribe audio for a single packet using the OVQwen3ASR pipeline."""
        metrics = None
        final_text = ""

        try:
            qwen_cfg = Qwen3ASRConfig(
                audio_base64=packet.gen_config.audio_base64,
                language=packet.gen_config.language,
                max_tokens=packet.gen_config.max_tokens,
                max_chunk_sec=packet.gen_config.max_chunk_sec,
                search_expand_sec=packet.gen_config.search_expand_sec,
                min_window_ms=packet.gen_config.min_window_ms,
            )
            async for item in asr_model.transcribe(qwen_cfg):
                if isinstance(item, dict):
                    metrics = item
                else:
                    final_text = item

            packet.response = final_text
            packet.metrics = metrics
        except Exception as e:
            logger.error("Qwen3 ASR inference failed!", exc_info=True)
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
    
    """
    
    @staticmethod
    async def queue_worker_llm(model_name: str, model_queue: asyncio.Queue, llm_model: OVGenAI_LLM, registry: ModelRegistry):
        """Text model inference worker that processes packets from queue"""
        logger.info(f"[LLM Worker: {model_name}] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                logger.info(f"[LLM Worker: {model_name}] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_llm(packet, llm_model)

            # Check if inference failed and trigger model unload
            if completed_packet.response and completed_packet.response.startswith("Error:"):
                logger.error(f"[LLM Worker: {model_name}] Inference failed, triggering model unload...")
                asyncio.create_task(registry.register_unload(model_name))
                break

            if completed_packet.metrics:
                logger.info(f"[LLM Worker: {model_name}] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    async def queue_worker_vlm(model_name: str, model_queue: asyncio.Queue, vlm_model: OVGenAI_VLM, registry: ModelRegistry):
        """Image model inference worker that processes packets from queue"""
        logger.info(f"[VLM Worker: {model_name}] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                logger.info(f"[VLM Worker: {model_name}] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_vlm(packet, vlm_model)

            # Check if inference failed and trigger model unload
            if completed_packet.response and completed_packet.response.startswith("Error:"):
                logger.error(f"[VLM Worker: {model_name}] Inference failed, triggering model unload...")
                asyncio.create_task(registry.register_unload(model_name))
                break

            if completed_packet.metrics:
                logger.info(f"[VLM Worker: {model_name}] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    async def queue_worker_whisper(model_name: str, model_queue: asyncio.Queue, whisper_model: OVGenAI_Whisper, registry: ModelRegistry):
        """Whisper model inference worker that processes packets from queue"""
        logger.info(f"[Whisper Worker: {model_name}] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                logger.info(f"[Whisper Worker: {model_name}] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_whisper(packet, whisper_model)

            # Check if inference failed and trigger model unload
            if completed_packet.response and completed_packet.response.startswith("Error:"):
                logger.error(f"[Whisper Worker: {model_name}] Inference failed, triggering model unload...")
                asyncio.create_task(registry.register_unload(model_name))
                break

            if completed_packet.metrics:
                logger.info(f"[Whisper Worker: {model_name}] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    async def queue_worker_qwen3_asr(model_name: str, model_queue: asyncio.Queue, asr_model: OVQwen3ASR, registry: ModelRegistry):
        """Qwen3 ASR model inference worker that processes packets from queue."""
        logger.info(f"[Qwen3ASR Worker: {model_name}] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                logger.info(f"[Qwen3ASR Worker: {model_name}] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_qwen3_asr(packet, asr_model)

            if completed_packet.response and completed_packet.response.startswith("Error:"):
                logger.error(f"[Qwen3ASR Worker: {model_name}] Inference failed, triggering model unload...")
                asyncio.create_task(registry.register_unload(model_name))
                break

            if completed_packet.metrics:
                logger.info(f"[Qwen3ASR Worker: {model_name}] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    async def queue_worker_kokoro(model_name: str, model_queue: asyncio.Queue, kokoro_model: OV_Kokoro, registry: ModelRegistry):
        """Kokoro model inference worker that processes packets from queue"""
        logger.info(f"[Kokoro Worker: {model_name}] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                logger.info(f"[Kokoro Worker: {model_name}] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_kokoro(packet, kokoro_model)

            # Check if inference failed and trigger model unload
            if completed_packet.response and completed_packet.response.startswith("Error:"):
                logger.error(f"[Kokoro Worker: {model_name}] Inference failed, triggering model unload...")
                asyncio.create_task(registry.register_unload(model_name))
                break

            # Log the text that was converted to speech
            
            if completed_packet.metrics:
                logger.info(f"[Kokoro Worker: {model_name}] Metrics: {completed_packet.metrics}")

            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)

            model_queue.task_done()

    @staticmethod
    async def queue_worker_emb(model_name: str, model_queue: asyncio.Queue, emb_model: Optimum_EMB, registry: ModelRegistry):
        """EMB model inference worker that processes packets from queue"""
        logger.info(f"[EMB Worker: {model_name}] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                logger.info(f"[EMB Worker: {model_name}] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_emb(packet, emb_model)
            # Check if inference failed and trigger model unload
            if not completed_packet.response:
                logger.error(f"[EMB Worker: {model_name}] Inference failed, triggering model unload...")
                asyncio.create_task(registry.register_unload(model_name))
                break
            if completed_packet.metrics:
                logger.info(f"[EMB Worker: {model_name}] Metrics: {completed_packet.metrics}")
            if packet.result_future is not None and not packet.result_future.done():
                packet.result_future.set_result(completed_packet)
            model_queue.task_done()

    @staticmethod
    async def queue_worker_rr(model_name: str, model_queue: asyncio.Queue, rr_model: Optimum_RR, registry: ModelRegistry):
        """Reranker model inference worker that processes packets from queue"""
        logger.info(f"[Reranker Worker: {model_name}] Started, waiting for packets...")
        while True:
            packet = await model_queue.get()
            if packet is None:
                logger.info(f"[Reranker Worker: {model_name}] Shutdown signal received.")
                break

            completed_packet = await InferWorker.infer_rerank(packet, rr_model)
            # Check if inference failed and trigger model unload
            if not completed_packet.response:
                logger.error(f"[Reranker Worker: {model_name}] Inference failed, triggering model unload...")
                asyncio.create_task(registry.register_unload(model_name))
                break
            if completed_packet.metrics:
                logger.info(f"[Reranker Worker: {model_name}] Metrics: {completed_packet.metrics}")
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

        self._model_queues_qwen3_asr: Dict[str, asyncio.Queue] = {}
        self._model_tasks_qwen3_asr: Dict[str, asyncio.Task] = {}

        self._model_queues_kokoro: Dict[str, asyncio.Queue] = {}
        self._model_tasks_kokoro: Dict[str, asyncio.Task] = {}
        
        self._model_queues_emb: Dict[str, asyncio.Queue] = {}
        self._model_tasks_emb: Dict[str, asyncio.Task] = {}

        self._model_queues_rerank: Dict[str, asyncio.Queue] = {}
        self._model_tasks_rerank: Dict[str, asyncio.Task] = {}

        self._lock = asyncio.Lock()
        
        # Track active requests for cancellation: request_id -> (model_name, packet)
        self._active_requests: Dict[str, tuple[str, WorkerPacket]] = {}

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
                    task = asyncio.create_task(QueueWorker.queue_worker_llm(record.model_name, q, instance, self._model_registry))
                    self._model_tasks_llm[record.model_name] = task

            elif mt == ModelType.VLM and isinstance(instance, OVGenAI_VLM):
                if record.model_name not in self._model_queues_vlm:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_vlm[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_vlm(record.model_name, q, instance, self._model_registry))
                    self._model_tasks_vlm[record.model_name] = task

            elif mt == ModelType.WHISPER and isinstance(instance, OVGenAI_Whisper):
                if record.model_name not in self._model_queues_whisper:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_whisper[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_whisper(record.model_name, q, instance, self._model_registry))
                    self._model_tasks_whisper[record.model_name] = task

            elif mt == ModelType.QWEN3_ASR and isinstance(instance, OVQwen3ASR):
                if record.model_name not in self._model_queues_qwen3_asr:
                    q = asyncio.Queue()
                    self._model_queues_qwen3_asr[record.model_name] = q
                    task = asyncio.create_task(
                        QueueWorker.queue_worker_qwen3_asr(record.model_name, q, instance, self._model_registry)
                    )
                    self._model_tasks_qwen3_asr[record.model_name] = task

            elif mt == ModelType.KOKORO and isinstance(instance, OV_Kokoro):
                if record.model_name not in self._model_queues_kokoro:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_kokoro[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_kokoro(record.model_name, q, instance, self._model_registry))
                    self._model_tasks_kokoro[record.model_name] = task

            elif mt == ModelType.EMB and isinstance(instance, Optimum_EMB):
                if record.model_name not in self._model_queues_emb:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_emb[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_emb(record.model_name, q, instance, self._model_registry))
                    self._model_tasks_emb[record.model_name] = task
            
            elif mt == ModelType.RERANK and isinstance(instance, Optimum_RR):
                if record.model_name not in self._model_queues_rerank:
                    q: asyncio.Queue = asyncio.Queue()
                    self._model_queues_rerank[record.model_name] = q
                    task = asyncio.create_task(QueueWorker.queue_worker_rr(record.model_name, q, instance, self._model_registry))
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

            # Try qwen3_asr dicts
            q = self._model_queues_qwen3_asr.pop(record.model_name, None)
            t = self._model_tasks_qwen3_asr.pop(record.model_name, None)
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

    def _get_qwen3_asr_queue(self, model_name: str) -> asyncio.Queue:
        q = self._model_queues_qwen3_asr.get(model_name)
        if q is not None:
            return q
        raise ValueError(f"Qwen3 ASR model '{model_name}' is not loaded or no worker is available")

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
        gen_config.request_id = request_id  # Set request_id for cancellation tracking
        
        stream_queue: asyncio.Queue = asyncio.Queue()
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        packet = WorkerPacket(
            request_id=request_id,
            id_model=model_name,
            gen_config=gen_config,
            stream_queue=stream_queue,
            result_future=result_future,
        )
        
        # Register active request
        async with self._lock:
            self._active_requests[request_id] = (model_name, packet)
        
        try:
            q = self._get_model_queue(model_name)
            await q.put(packet)
            while True:
                item = await stream_queue.get()
                if item is None:
                    break
                yield item
        finally:
            # Unregister active request when done
            async with self._lock:
                self._active_requests.pop(request_id, None)

    async def infer_cancel(self, request_id: str) -> bool:
        """
        Cancel an ongoing inference request by request_id.
        
        Args:
            request_id: The request ID to cancel
            
        Returns:
            True if cancellation was triggered, False if request not found
        """
        async with self._lock:
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

    async def transcribe_qwen3_asr(self, model_name: str, gen_config: OV_Qwen3ASRGenConfig) -> Dict[str, Any]:
        """Transcribe audio using Qwen3 ASR model."""
        request_id = uuid.uuid4().hex
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()
        packet = WorkerPacket(
            request_id=request_id,
            id_model=model_name,
            gen_config=gen_config,
            result_future=result_future,
        )
        q = self._get_qwen3_asr_queue(model_name)
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