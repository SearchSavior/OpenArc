"""
RemoteOVGenAI_VLM -- in-process facade for the out-of-process VLM worker.

Presents the same async surface the server calls on a VLM instance
(load_model / generate_type / cancel / unload_model), but every OpenVINO call
happens in a supervised child process:

  * unload = terminate the process (no poisoned Core can be left behind)
  * reload = spawn a fresh process (fresh ov::Core)
  * wedge  = the worker dies or reports FATAL; the supervisor respawns a
             fresh process within a per-load budget, and once the budget is
             exhausted the model is unloaded from the registry (readiness
             drops, an operator reloads with a fresh budget)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional, Union

from src.engine.worker import protocol as proto
from src.engine.worker.supervisor import EOF, WorkerSupervisor
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import ModelLoadConfig

if TYPE_CHECKING:
    from src.server.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class RemoteOVGenAI_VLM:
    """Drop-in replacement for OVGenAI_VLM that runs the pipeline in a child process."""

    def __init__(
        self,
        load_config: ModelLoadConfig,
        *,
        registry: Optional["ModelRegistry"] = None,
        max_respawns: int = 2,
        load_timeout: Optional[float] = None,
    ) -> None:
        self.load_config = load_config
        self.model_name = load_config.model_name
        self._registry = registry
        self._supervisor = WorkerSupervisor(
            self.model_name,
            max_respawns=max_respawns,
            load_timeout=load_timeout,
            on_dead=self._on_worker_dead,
        )

    # -- registry-facing surface ------------------------------------------------
    async def load_model(self, loader: ModelLoadConfig) -> None:
        """Spawn the worker process and load the pipeline inside it."""
        logger.info(f"[{loader.model_name}] loading model in a dedicated worker process")
        await self._supervisor.start(loader)

    async def unload_model(self, registry: "ModelRegistry", model_name: str) -> bool:
        """Unload: unregister from the registry and terminate the worker process."""
        self._registry = registry
        removed = await registry.register_unload(model_name)
        try:
            await self._supervisor.unload()
            logger.info(f"[{model_name}] inference worker terminated; model unloaded")
        except Exception as e:
            logger.warning(f"[{model_name}] error while terminating inference worker: {e}")
        return removed

    async def _on_worker_dead(self) -> None:
        """The supervisor used up its respawn budget: drop the model."""
        logger.error(
            f"[{self.model_name}] inference worker is permanently dead "
            f"(respawn budget exhausted); unloading the model"
        )
        if self._registry is not None:
            await self._registry.register_unload(self.model_name)

    # -- inference surface (called by WorkerRegistry) -----------------------------
    def generate_type(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """Unified generation; same yield contract as OVGenAI_VLM.generate_type."""
        return self._generate(gen_config)

    async def _generate(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        queue, result = await self._supervisor.begin_generate(
            gen_config.model_dump_json(), gen_config.request_id
        )
        try:
            while True:
                item = await queue.get()
                if item is EOF:
                    break
                yield item
            await result  # raises if the worker reported an error
        except GeneratorExit:
            # The consumer stopped listening (e.g. client disconnect): ask
            # the worker to stop producing instead of burning GPU for no one.
            if gen_config.request_id is not None:
                try:
                    await self._supervisor.request_cancel(gen_config.request_id)
                except Exception:
                    pass
            raise
        finally:
            if not result.done():
                result.cancel()

    async def cancel(self, request_id: str) -> bool:
        """Cancel an ongoing streaming generation by request_id."""
        return await self._supervisor.request_cancel(request_id)

    # -- introspection ----------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return self._supervisor.status()

    @property
    def worker_pid(self) -> Optional[int]:
        return self._supervisor.pid
