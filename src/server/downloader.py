import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_FORBIDDEN_PATHS = frozenset({
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64",
    "/proc", "/run", "/sbin", "/sys", "/usr", "/var",
})


def validate_download_path(path: str) -> str:
    """Validate and resolve a download path. Raises ValueError if unsafe."""
    resolved = Path(path).resolve()

    if resolved.parent == resolved:
        raise ValueError(f"Cannot download to filesystem root: {resolved}")

    if str(resolved) in _FORBIDDEN_PATHS:
        raise ValueError(f"Cannot download to system directory: {resolved}")

    resolved.mkdir(parents=True, exist_ok=True)

    if not resolved.is_dir():
        raise ValueError(f"Path is not a directory: {resolved}")

    if not os.access(resolved, os.W_OK):
        raise ValueError(f"Path is not writable: {resolved}")

    return str(resolved)


class DownloadTask:
    def __init__(self, model_name: str, path: Optional[str] = None):
        self.model_name = model_name
        self.path = path
        self.status = "pending"  # pending, downloading, paused, cancelled, completed, error
        self.task: Optional[asyncio.Task] = None
        self.progress = 0.0
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self.total_files = 0
        self.completed_files = 0
        self.error_msg = ""
        self.started_at = time.time()
        self.completed_at: Optional[float] = None

        # Thread-safe control signals
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self._pause.set()  # Not paused by default


class Downloader:
    CLEANUP_AFTER_SECONDS = 3600

    def __init__(self):
        self.tasks: Dict[str, DownloadTask] = {}
        self._lock = asyncio.Lock()

    async def start(self, model_name: str, path: Optional[str] = None) -> bool:
        """Start a new download. Returns False if one is already active."""
        if path:
            path = validate_download_path(path)

        async with self._lock:
            self._cleanup_stale()

            existing = self.tasks.get(model_name)
            if existing and existing.status in ("downloading", "paused"):
                return False

            task = DownloadTask(model_name, path)
            self.tasks[model_name] = task
            task.status = "downloading"
            task.task = asyncio.create_task(self._download_worker(task))
            return True

    def pause(self, model_name: str) -> bool:
        """Pause an active download. The worker thread blocks between files."""
        task = self.tasks.get(model_name)
        if not task or task.status != "downloading":
            return False
        task.status = "paused"
        task._pause.clear()
        return True

    def resume(self, model_name: str) -> bool:
        """Resume a paused download."""
        task = self.tasks.get(model_name)
        if not task or task.status != "paused":
            return False
        task.status = "downloading"
        task._pause.set()
        return True

    def cancel(self, model_name: str) -> bool:
        """Cancel an active or paused download."""
        task = self.tasks.get(model_name)
        if not task or task.status in ("completed", "cancelled"):
            return False
        task.status = "cancelled"
        task.completed_at = time.time()
        task._cancel.set()
        task._pause.set()  # Unblock thread so it can exit
        if task.task:
            task.task.cancel()
        return True

    def list_tasks(self) -> List[Dict]:
        self._cleanup_stale()
        return [
            {
                "model_name": t.model_name,
                "status": t.status,
                "progress": round(t.progress, 2),
                "total_bytes": t.total_bytes,
                "downloaded_bytes": t.downloaded_bytes,
                "total_files": t.total_files,
                "completed_files": t.completed_files,
                "error": t.error_msg,
                "started_at": t.started_at,
                "completed_at": t.completed_at,
            }
            for t in self.tasks.values()
        ]

    def _cleanup_stale(self):
        """Remove terminal tasks older than CLEANUP_AFTER_SECONDS."""
        now = time.time()
        stale = [
            name
            for name, t in self.tasks.items()
            if t.status in ("completed", "cancelled", "error")
            and t.completed_at
            and (now - t.completed_at) > self.CLEANUP_AFTER_SECONDS
        ]
        for name in stale:
            del self.tasks[name]

    async def _download_worker(self, task: DownloadTask):
        try:
            await asyncio.to_thread(self._download_sync, task)
            if not task._cancel.is_set():
                task.status = "completed"
                task.progress = 100.0
                task.completed_at = time.time()
                logger.info(f"Download completed for {task.model_name}")
        except asyncio.CancelledError:
            logger.info(f"Download {task.status} for {task.model_name}")
        except Exception as e:
            task.status = "error"
            task.error_msg = str(e)
            task.completed_at = time.time()
            logger.error(f"Download failed for {task.model_name}: {e}")

    def _download_sync(self, task: DownloadTask):
        """Blocking download in a worker thread.

        Downloads files individually via hf_hub_download so we can report
        per-file progress and honour pause/cancel signals between files.
        """
        import huggingface_hub

        api = huggingface_hub.HfApi()
        repo = api.repo_info(task.model_name, files_metadata=True)
        siblings = repo.siblings or []

        files = [(f.rfilename, getattr(f, "size", None) or 0) for f in siblings]
        task.total_bytes = sum(size for _, size in files)
        task.total_files = len(files)

        if task._cancel.is_set():
            return

        for filename, file_size in files:
            if task._cancel.is_set():
                return

            # Block here while paused
            task._pause.wait()

            if task._cancel.is_set():
                return

            huggingface_hub.hf_hub_download(
                repo_id=task.model_name,
                filename=filename,
                local_dir=task.path,
            )

            task.completed_files += 1
            task.downloaded_bytes += file_size
            if task.total_bytes > 0:
                task.progress = (task.downloaded_bytes / task.total_bytes) * 100.0


global_downloader = Downloader()
