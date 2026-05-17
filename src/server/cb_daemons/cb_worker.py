from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.engine.ov_genai.continuous_batching.cb_adapter_llm import ArcCBLLM
    from src.server.models.ov_genai import OVGenAI_GenConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class CBRequest:
    """
    Self-contained CB request object.

    Deliberately independent of WorkerPacket / WorkerRegistry: the only thing
    the route consumes is the yield shape pushed onto `stream_queue`
    (text str, then {"metrics": ...}, terminated by None).
    """
    gen_config: OVGenAI_GenConfig
    stream_queue: asyncio.Queue
    int_id: int = -1


@dataclass
class _ActiveState:
    request: CBRequest
    handle: Any
    n_input_tokens: int
    generated_ids: List[int] = field(default_factory=list)
    last_print_len: int = 0          # length of decoded text already emitted
    emitted_tokens: int = 0          # number of generated ids already emitted


class CBInferDaemon:
    """
    Owns one model's ArcCBLLM, an admission queue, the active request map, and
    a single-thread executor that is the serialization boundary for ALL
    pipeline/handle calls (ContinuousBatchingPipeline is single-thread-owned).
    """

    def __init__(self, model_name: str, arc: ArcCBLLM):
        self.model_name = model_name
        self.arc = arc
        self._admit: asyncio.Queue = asyncio.Queue()
        self._active: Dict[int, _ActiveState] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"cb-{model_name}")
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def submit(self, request: CBRequest) -> None:
        await self._admit.put(request)

    async def stop(self) -> None:
        self._stopping = True
        # Wake the loop so it can observe _stopping.
        await self._admit.put(None)
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10)
            except asyncio.TimeoutError:
                self._task.cancel()
            except Exception:
                # Shutdown must not propagate a crashed/cancelled loop.
                logger.error(
                    f"[CBInferDaemon: {self.model_name}] loop ended with error",
                    exc_info=True,
                )
        # Terminate any still-open streams so HTTP responses do not hang.
        for st in list(self._active.values()):
            await self._safe_put(st.request, None)
        self._active.clear()
        self._executor.shutdown(wait=False)

    async def _run(self) -> None:
        import openvino_genai as genai

        running_status = genai.GenerationStatus.RUNNING
        loop = asyncio.get_running_loop()
        logger.info(f"[CBInferDaemon: {self.model_name}] started")
        try:
            while not self._stopping:
                if not self._active:
                    # Idle: block until something is admitted (no busy spin).
                    req = await self._admit.get()
                    if req is None:
                        continue
                    await self._admit_request(loop, req)

                # Drain any other pending admissions without blocking.
                while not self._admit.empty():
                    req = self._admit.get_nowait()
                    if req is None:
                        continue
                    await self._admit_request(loop, req)

                if not self._active:
                    continue

                await loop.run_in_executor(self._executor, self.arc.step)

                finished: List[int] = []
                for int_id, st in list(self._active.items()):
                    try:
                        can_read = await loop.run_in_executor(
                            self._executor, st.handle.can_read
                        )
                        if can_read:
                            outputs = await loop.run_in_executor(
                                self._executor, st.handle.read
                            )
                            for out in outputs.values():
                                st.generated_ids.extend(out.generated_ids)
                            await self._emit(st, final=False)

                        status = await loop.run_in_executor(
                            self._executor, st.handle.get_status
                        )
                        if status != running_status:
                            finished.append(int_id)
                    except Exception:
                        logger.error(
                            f"[CBInferDaemon: {self.model_name}] request {int_id} failed",
                            exc_info=True,
                        )
                        await self._safe_put(st.request, None)
                        self._active.pop(int_id, None)

                for int_id in finished:
                    st = self._active.pop(int_id, None)
                    if st is not None:
                        await self._finalize(st)
        except asyncio.CancelledError:
            raise
        finally:
            logger.info(f"[CBInferDaemon: {self.model_name}] stopped")

    async def _admit_request(self, loop: asyncio.AbstractEventLoop, req: CBRequest) -> None:
        try:
            handle, n_input = await loop.run_in_executor(
                self._executor, self.arc.add_request, req.int_id, req.gen_config
            )
        except Exception:
            logger.error(
                f"[CBInferDaemon: {self.model_name}] add_request failed for {req.int_id}",
                exc_info=True,
            )
            await self._safe_put(req, None)
            return
        self._active[req.int_id] = _ActiveState(
            request=req, handle=handle, n_input_tokens=n_input
        )

    async def _emit(self, st: _ActiveState, final: bool) -> None:
        """
        Accumulate-then-delta decode. `generated_ids` is the cumulative buffer
        (each handle.read() yields only NEW ids, extended in the step loop).
        We decode the whole buffer and emit only the text past last_print_len.
        Non-final emits are gated on stream_chunk_tokens; final always flushes.
        """
        total_tokens = len(st.generated_ids)
        if total_tokens == 0:
            return
        if not final:
            chunk_tokens = max(1, getattr(st.request.gen_config, "stream_chunk_tokens", 1))
            if (total_tokens - st.emitted_tokens) < chunk_tokens:
                return

        text = self.arc.decode(st.generated_ids)
        if len(text) > st.last_print_len:
            delta = text[st.last_print_len:]
            if delta:
                await self._safe_put(st.request, delta)
            st.last_print_len = len(text)
        st.emitted_tokens = total_tokens

    async def _finalize(self, st: _ActiveState) -> None:
        # Final flush of any remaining decoded text.
        await self._emit(st, final=True)
        metrics = self.arc.collect_metrics(
            input_token=st.n_input_tokens,
            new_token=len(st.generated_ids),
        )
        await self._safe_put(st.request, {"metrics": metrics})
        await self._safe_put(st.request, None)

    @staticmethod
    async def _safe_put(req: CBRequest, item: Any) -> None:
        try:
            await req.stream_queue.put(item)
        except Exception:
            logger.error("CB stream_queue put failed", exc_info=True)
