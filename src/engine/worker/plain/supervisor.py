"""
Supervisor for the plain-OpenVINO inference workers (stage 3).

Identical to the GenAI WorkerSupervisor -- same lifecycle, watchdog,
respawn budget, and unload ladder -- except the child it spawns speaks the
plain-OpenVINO worker protocol (src.engine.worker.plain.worker_process).
"""

from __future__ import annotations

from src.engine.worker.supervisor import WorkerSupervisor


class PlainWorkerSupervisor(WorkerSupervisor):
    """WorkerSupervisor for the plain-OpenVINO worker entry point."""

    WORKER_ENTRY = "src.engine.worker.plain.worker_process"
