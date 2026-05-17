from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional, Union

from src.server.cb_daemons.cb_worker import CBInferDaemon, CBRequest
from src.server.model_registry import ModelRecord, ModelRegistry
from src.server.models.registration import ModelType

if TYPE_CHECKING:
    from src.server.models.ov_genai import OVGenAI_GenConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CBRequestDaemon:
    """
    Thin admission front-end: assigns a monotonically increasing internal int
    request id and forwards to the model's CBInferDaemon.
    """

    def __init__(self, model_name: str, infer: CBInferDaemon):
        self.model_name = model_name
        self._infer = infer
        self._admit: asyncio.Queue = asyncio.Queue()
        self._counter = 0
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def submit(self, request: CBRequest) -> None:
        await self._admit.put(request)

    async def stop(self) -> None:
        self._stopping = True
        await self._admit.put(None)
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10)
            except asyncio.TimeoutError:
                self._task.cancel()
            except Exception:
                logger.error(
                    f"[CBRequestDaemon: {self.model_name}] loop ended with error",
                    exc_info=True,
                )

    async def _run(self) -> None:
        logger.info(f"[CBRequestDaemon: {self.model_name}] started")
        try:
            while not self._stopping:
                request = await self._admit.get()
                if request is None:
                    continue
                self._counter += 1
                request.int_id = self._counter
                await self._infer.submit(request)
        except asyncio.CancelledError:
            raise
        finally:
            logger.info(f"[CBRequestDaemon: {self.model_name}] stopped")


class CBRouter:
    """
    Standalone continuous-batching dispatcher. Subscribes its own load/unload
    callbacks to the shared ModelRegistry and owns per-model daemons. Does NOT
    touch WorkerRegistry.
    """

    def __init__(self, model_registry: ModelRegistry):
        self._model_registry = model_registry
        self._request_daemons: Dict[str, CBRequestDaemon] = {}
        self._infer_daemons: Dict[str, CBInferDaemon] = {}
        self._lock = asyncio.Lock()
        self._model_registry.add_on_loaded(self._on_model_loaded)
        self._model_registry.add_on_unloaded(self._on_model_unloaded)

    @staticmethod
    def _normalize_model_type(mt) -> Optional[ModelType]:
        if isinstance(mt, ModelType):
            return mt
        try:
            return ModelType(mt)
        except Exception:
            return None

    def is_cb_model(self, model_name: str) -> bool:
        return model_name in self._request_daemons

    async def _on_model_loaded(self, record: ModelRecord) -> None:
        mt = self._normalize_model_type(record.model_type)
        if mt != ModelType.CB_LLM:
            return
        instance = record.model_instance
        # Duck-typed adapter check (avoids importing the heavy ArcCBLLM module here).
        if instance is None or not all(
            callable(getattr(instance, m, None))
            for m in ("add_request", "step", "has_non_finished_requests", "decode", "collect_metrics")
        ):
            logger.info(
                f"[CBRouter] cb_llm instance unusable for {record.model_name}: {type(instance)}"
            )
            return
        async with self._lock:
            if record.model_name in self._request_daemons:
                return
            infer = CBInferDaemon(record.model_name, instance)
            request_daemon = CBRequestDaemon(record.model_name, infer)
            infer.start()
            request_daemon.start()
            self._infer_daemons[record.model_name] = infer
            self._request_daemons[record.model_name] = request_daemon
            logger.info(f"[CBRouter] started CB daemons for {record.model_name}")

    async def _on_model_unloaded(self, record: ModelRecord) -> None:
        async with self._lock:
            request_daemon = self._request_daemons.pop(record.model_name, None)
            infer = self._infer_daemons.pop(record.model_name, None)
        if request_daemon is not None:
            await request_daemon.stop()
        if infer is not None:
            await infer.stop()
        if request_daemon is not None or infer is not None:
            logger.info(f"[CBRouter] stopped CB daemons for {record.model_name}")

    async def stream_generate(
        self, model_name: str, gen_config: OVGenAI_GenConfig
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Yields decoded text chunks, then {"metrics": ...}; terminates internally
        on the None sentinel. Drop-in for the route's existing async-for loop.
        """
        request_daemon = self._request_daemons.get(model_name)
        if request_daemon is None:
            raise ValueError(
                f"Continuous batching model '{model_name}' is not loaded"
            )

        request = CBRequest(gen_config=gen_config, stream_queue=asyncio.Queue())
        await request_daemon.submit(request)
        while True:
            item = await request.stream_queue.get()
            if item is None:
                break
            yield item
