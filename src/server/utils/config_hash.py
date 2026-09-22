"""Model config hashing: detect config changes and recompile on demand.

Every model entry in ``openarc_config.json`` carries a ``config_hash``: the
SHA-256 of the (path-resolved) load configuration the model was last
*compiled* with.

Why this exists: OpenVINO caches compiled model blobs in the model's
``cache_dir`` (see ``ModelLoadConfig.cache_dir``) and keys that cache by the
model *files* and the device -- **not** by the runtime/scheduler configuration.
So when a model entry changes (``--runtime-config``, ``--scheduler-config``,
device, ``cache_dir``, a re-run of ``openarc add``, a manual edit of
``openarc_config.json``, ...) a subsequent load can pick up blobs that were
compiled under the old settings. The stale graph then conflicts with the new
configuration -- load failures (e.g. ``CL_INVALID_EVENT`` / program-builder
errors) or inference with the wrong limits.

The rule applied by :func:`check_model_config_hash` at every model load:

1. Hash the configuration the model is about to be loaded with (the resolved
   ``ModelLoadConfig``, after the context window has been resolved, since the
   resolved window also reaches the compiled pipeline).
2. Look up the model's entry in ``openarc_config.json``.
3. If the stored ``config_hash`` differs from the current one -- including a
   missing stored hash, which is exactly what a freshly rewritten entry from
   ``openarc add`` looks like -- the compiled-model cache is invalidated so
   the pipeline is recompiled with the new settings.
4. The current hash is written back to the config file (unless the model is
   not in the file at all, e.g. a raw ``POST /openarc/load`` with no entry,
   in which case there is nothing to track and the cache is left alone so
   cache-less API users keep their cache reuse).

Hashing the *resolved* loader (absolute paths) keeps the value stable across
both load entry points -- server startup (:mod:`src.server.main`) and the
``openarc load`` CLI (``POST /openarc/load``) -- which resolve relative
``model_path`` / ``draft_model_path`` / ``cache_dir`` against the config
file's directory before loading.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

from src.cli.modules.server_config import ServerConfig
from src.server.schemas.registration import ModelLoadConfig

logger = logging.getLogger(__name__)

# Key under which the hash is stored in the model's openarc_config.json entry.
CONFIG_HASH_KEY = "config_hash"


def compute_config_hash(load_config: ModelLoadConfig) -> str:
    """SHA-256 of a load configuration's canonical form.

    The canonical form is the JSON serialization of the *resolved*
    ``ModelLoadConfig`` (all fields, defaults included, keys sorted), with the
    :data:`CONFIG_HASH_KEY` field itself excluded so the hash never folds into
    its own input. Two loads of the same logical config -- regardless of the
    key order in the JSON on the wire -- produce the same hash; any change to
    a configuration field (``runtime_config``, ``scheduler_config``, device,
    ``cache_dir``, the resolved ``context_window``, ...) changes it.

    The result is prefixed with the algorithm (``sha256:``) so the value is
    self-describing if the scheme ever changes.
    """
    payload = load_config.model_dump(mode="json", exclude={CONFIG_HASH_KEY})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def invalidate_compiled_model_cache(cache_dir: Optional[str]) -> bool:
    """Delete a model's OpenVINO compiled-model cache so the next load recompiles.

    ``cache_dir`` is the directory passed to OpenVINO as ``CACHE_DIR``; it
    holds the compiled model blobs (``*.blob``) and OpenCL kernel caches
    (``*.cl_cache``) from previous loads. Deleting the whole directory is the
    documented way to force a recompile (see docs/models.md), and it is safe
    for models that are currently loaded: their graphs already live in memory,
    the directory only affects future compiles.

    Best effort: a missing directory is a no-op, and a failure (e.g. a
    read-only filesystem) is logged, not raised -- the load proceeds and
    OpenVINO may still find a stale entry, which is no worse than the status
    quo.

    Returns:
        True if files were deleted, False if there was nothing to delete (or
        deletion failed).
    """
    if not cache_dir:
        return False
    path = Path(cache_dir)
    try:
        if path.is_file():
            path.unlink()
            return True
        if not path.is_dir():
            return False
        shutil.rmtree(path)
        return True
    except OSError as e:
        logger.error(f"Failed to invalidate compiled-model cache at {cache_dir}: {e}")
        return False


def _model_config_entry(model_name: str) -> Optional[dict]:
    """The model's raw entry from openarc_config.json, or None if absent."""
    config = ServerConfig().load_config()
    models = config.get("models")
    if not isinstance(models, dict):
        return None
    entry = models.get(model_name)
    return entry if isinstance(entry, dict) else None


def persist_model_config_hash(model_name: str, config_hash: str) -> bool:
    """Write ``config_hash`` into the model's openarc_config.json entry.

    Returns:
        True when the entry already carried the hash or it was written, False
        when the model has no entry in the config file (nothing to persist).
    """
    config_manager = ServerConfig()
    config = config_manager.load_config()
    models = config.get("models")
    if not isinstance(models, dict) or not isinstance(models.get(model_name), dict):
        return False
    if models[model_name].get(CONFIG_HASH_KEY) == config_hash:
        return True  # already up to date; skip the write (read-only fs friendly)
    models[model_name][CONFIG_HASH_KEY] = config_hash
    config["models"] = models
    config_manager.save_config(config)
    return True


def check_model_config_hash(
    load_config: ModelLoadConfig, *, force_recompile: bool = False
) -> Tuple[str, bool]:
    """Config-change gate run at every model load.

    Computes the hash of the configuration about to be compiled, compares it
    against the hash stored in the model's ``openarc_config.json`` entry,
    invalidates the compiled-model cache on a difference (triggering a
    recompile), and persists the current hash.

    Args:
        load_config: The RESOLVED load configuration (context window already
            resolved, paths already absolute), as handed to the engine.
        force_recompile: When True, the compiled-model cache is invalidated and the
            pipeline is forced to recompile from the IR even though the configuration
            is UNCHANGED (so the config-hash gate would stay closed). It is set by the
            ``--force-recompile`` / ``--fr`` operator flag on ``openarc serve start``
            and ``openarc load`` -- an explicit request to recompile rather than reuse
            the cache (e.g. after the operator swapped the model files or host, or
            simply to be sure the pipeline is freshly built). The recompile is a
            deliberate "full load"; it is deliberately NOT done on the ordinary
            (cache-warm) load path, where the fast cache load is wanted.

    Returns:
        (current_hash, recompiled): the hash of the configuration being
        compiled, and True when a config difference was detected OR
        ``force_recompile`` was requested, in which case the compiled-model cache
        invalidation was triggered (the cache may have been empty; the recompile
        is then simply a first compile).
    """
    current_hash = compute_config_hash(load_config)
    recompiled = False

    entry = _model_config_entry(load_config.model_name)
    if entry is not None:
        stored_hash = entry.get(CONFIG_HASH_KEY)
        config_changed = stored_hash != current_hash
        if config_changed or force_recompile:
            if config_changed:
                logger.info(
                    f"[{load_config.model_name}] Model config changed since the last "
                    f"compile (stored {CONFIG_HASH_KEY} "
                    f"{stored_hash!r} != current {current_hash!r}); invalidating the "
                    f"compiled-model cache so the pipeline recompiles with the new "
                    f"settings."
                )
            else:
                # Config is unchanged, but a recompile was forced anyway by the
                # operator --force-recompile / --fr. Invalidate the cache so the
                # pipeline rebuilds from the IR instead of re-importing its
                # existing (possibly stale) blobs.
                logger.info(
                    f"[{load_config.model_name}] Forcing a recompile; config is "
                    f"unchanged but the compiled-model cache {load_config.cache_dir!r} "
                    f"is invalidated so the pipeline rebuilds from the IR instead of "
                    f"re-importing its existing (possibly stale) blobs."
                )
            invalidate_compiled_model_cache(load_config.cache_dir)
            recompiled = True
        try:
            persist_model_config_hash(load_config.model_name, current_hash)
        except OSError as e:
            # A read-only config file must not break the load: the (critical)
            # invalidation already happened. The next load simply re-detects the
            # difference -- the re-invalidation is then a no-op, since the
            # cache directory is already gone -- and retries the write.
            logger.warning(
                f"[{load_config.model_name}] Could not persist {CONFIG_HASH_KEY} "
                f"to openarc_config.json: {e}"
            )
    else:
        # No entry in openarc_config.json (e.g. a raw POST /openarc/load).
        # Normally nothing is tracked and the cache is left as-is so cache-less
        # API users keep cache reuse. A forced recompile still has to clear the
        # cache so it recompiles fresh even for a model that is not tracked in the
        # config file -- the force came from an operator --force-recompile / --fr;
        # there is just no entry to persist the hash into.
        if force_recompile:
            invalidate_compiled_model_cache(load_config.cache_dir)
            recompiled = True
        logger.debug(
            f"[{load_config.model_name}] No entry in openarc_config.json; "
            f"skipping config-hash tracking "
            f"({'(forced recompile requested) forcing a recompile anyway' if force_recompile else 'cache left as-is'})."
        )

    return current_hash, recompiled
