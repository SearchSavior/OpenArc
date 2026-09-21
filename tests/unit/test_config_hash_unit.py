"""Unit tests for model config hashing and the recompile-on-change trigger.

Covers the compiled-model invalidation gate (src.server.utils.config_hash):
every model entry in openarc_config.json carries a config_hash of the
(path-resolved) configuration it was last compiled with. At every load the
current configuration is hashed and compared against the stored value; a
difference (changed --runtime-config / --scheduler-config, manual edit of
openarc_config.json, re-run of `openarc add`) invalidates the model's
compiled-model cache so the pipeline recompiles, and the new hash is written
back.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest  # type: ignore[import]

import src.server.model_registry as registry_module
from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import (
    EngineType,
    ModelLoadConfig,
    ModelType,
)
import src.server.utils.config_hash as config_hash_module
from src.server.utils.config_hash import (
    CONFIG_HASH_KEY,
    check_model_config_hash,
    compute_config_hash,
    invalidate_compiled_model_cache,
    persist_model_config_hash,
)


# --- fixtures / helpers ------------------------------------------------------


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point OPENARC_CONFIG_FILE at a temp config file."""
    path = tmp_path / "openarc_config.json"
    path.write_text(json.dumps({"models": {}}), encoding="utf-8")
    monkeypatch.setenv("OPENARC_CONFIG_FILE", str(path))
    return path


def _write_entry(config_file: Path, name: str, entry: dict) -> None:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config.setdefault("models", {})[name] = entry
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _read_entry(config_file: Path, name: str) -> dict:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    return config["models"][name]


def _load_config(
    name: str = "hash-model",
    model_path: str = "/models/mock",
    runtime_config: dict | None = None,
    cache_dir: str | None = "/tmp/openarc-cache/hash-model",
    device: str = "CPU",
    **kwargs,
) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_path=model_path,
        model_name=name,
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device=device,
        runtime_config=runtime_config if runtime_config is not None else {},
        cache_dir=cache_dir,
        **kwargs,
    )


def _make_cache_dir(tmp_path: Path, name: str) -> Path:
    cache_dir = tmp_path / "cache" / name
    cache_dir.mkdir(parents=True)
    (cache_dir / "12345.blob").write_bytes(b"compiled")
    (cache_dir / "67890.cl_cache").write_bytes(b"opencl")
    return cache_dir


def _reg(monkeypatch: pytest.MonkeyPatch, created: list) -> ModelRegistry:
    async def fake_create(config):  # type: ignore[override]
        created.append(config)
        return SimpleNamespace(
            unload_model=lambda *a, **k: None
        )

    monkeypatch.setattr(registry_module, "create_model_instance", fake_create)
    return ModelRegistry()


async def _await_unloaded(registry: ModelRegistry, name: str) -> None:
    """Wait until the background unload task has removed the record."""
    for _ in range(1000):
        async with registry._lock:
            present = any(r.model_name == name for r in registry._models.values())
        if not present:
            return
        await asyncio.sleep(0)
    raise TimeoutError(f"model '{name}' was not unloaded in time")


# --- compute_config_hash -----------------------------------------------------


def test_config_hash_is_stable_for_equivalent_configs() -> None:
    a = _load_config(runtime_config={"OFFLOAD_RATIO": 0.05, "NUM_STREAMS": 2})
    b = _load_config(runtime_config={"NUM_STREAMS": 2, "OFFLOAD_RATIO": 0.05})
    # Omitted optional fields default to the same values as explicit None.
    c = _load_config(
        runtime_config={"OFFLOAD_RATIO": 0.05, "NUM_STREAMS": 2},
        max_tokens=None,
        context_window=None,
    )
    assert compute_config_hash(a) == compute_config_hash(b)
    assert compute_config_hash(a) == compute_config_hash(c)
    assert compute_config_hash(a).startswith("sha256:")
    assert len(compute_config_hash(a)) == len("sha256:") + 64


def test_config_hash_changes_with_runtime_and_scheduler_config() -> None:
    base = compute_config_hash(_load_config())
    assert compute_config_hash(_load_config(runtime_config={"OFFLOAD_RATIO": 0.1})) != base
    assert (
        compute_config_hash(
            _load_config(scheduler_config={"max_num_batched_tokens": 128})
        )
        != base
    )
    assert (
        compute_config_hash(
            _load_config(scheduler_config={"max_num_batched_tokens": 128})
        )
        != compute_config_hash(
            _load_config(scheduler_config={"max_num_batched_tokens": 256})
        )
    )
    # device, context_window and cache_dir also reach the compiled pipeline.
    assert compute_config_hash(_load_config(device="GPU.0")) != base
    assert compute_config_hash(_load_config(context_window=4096)) != base
    assert compute_config_hash(_load_config(cache_dir="/other")) != base


def test_config_hash_ignores_the_hash_field_itself() -> None:
    a = _load_config()
    b = a.model_copy(update={"config_hash": compute_config_hash(a)})
    assert compute_config_hash(a) == compute_config_hash(b)


# --- invalidate_compiled_model_cache -----------------------------------------


def test_invalidate_deletes_cache_directory(tmp_path: Path) -> None:
    cache_dir = _make_cache_dir(tmp_path, "m")
    assert invalidate_compiled_model_cache(str(cache_dir)) is True
    assert not cache_dir.exists()


def test_invalidate_noop_when_missing_or_unset(tmp_path: Path) -> None:
    assert invalidate_compiled_model_cache(None) is False
    assert invalidate_compiled_model_cache(str(tmp_path / "does-not-exist")) is False


# --- persist_model_config_hash ------------------------------------------------


def test_persist_writes_hash_into_entry(config_file: Path) -> None:
    _write_entry(config_file, "m", {"model_name": "m", "device": "CPU"})
    assert persist_model_config_hash("m", "sha256:abc") is True
    assert _read_entry(config_file, "m")[CONFIG_HASH_KEY] == "sha256:abc"


def test_persist_noop_when_absent_or_unchanged(config_file: Path) -> None:
    # Model not in the config file: nothing to persist.
    assert persist_model_config_hash("ghost", "sha256:abc") is False
    assert json.loads(config_file.read_text(encoding="utf-8"))["models"] == {}

    _write_entry(config_file, "m", {"model_name": "m", CONFIG_HASH_KEY: "sha256:same"})
    before = config_file.read_text(encoding="utf-8")
    assert persist_model_config_hash("m", "sha256:same") is True
    assert config_file.read_text(encoding="utf-8") == before  # no rewrite


# --- check_model_config_hash (the gate) ---------------------------------------


def test_first_load_persists_hash_and_triggers_compile(config_file: Path, tmp_path: Path) -> None:
    """Entry exists but has no hash yet (fresh `openarc add`): the gate
    triggers a (first) compile and stores the hash."""
    cache_dir = _make_cache_dir(tmp_path, "first")
    entry = {
        "model_name": "first",
        "model_path": "/models/mock",
        "model_type": "llm",
        "engine": "ovgenai",
        "device": "CPU",
        "runtime_config": {},
        "cache_dir": str(cache_dir),
    }
    _write_entry(config_file, "first", entry)
    loader = _load_config(name="first", model_path="/models/mock", cache_dir=str(cache_dir))

    current_hash, recompiled = check_model_config_hash(loader)

    assert recompiled is True
    assert current_hash == compute_config_hash(loader)
    # Hash of the SAME logical config is stored, so the next unchanged load
    # (which resolves the identical paths) matches.
    assert _read_entry(config_file, "first")[CONFIG_HASH_KEY] == compute_config_hash(
        _load_config(name="first", model_path="/models/mock", cache_dir=str(cache_dir))
    )


def test_unchanged_load_keeps_cache_and_hash(config_file: Path, tmp_path: Path) -> None:
    cache_dir = _make_cache_dir(tmp_path, "same")
    _write_entry(
        config_file,
        "same",
        {"model_name": "same", "device": "CPU", "cache_dir": str(cache_dir)},
    )
    loader = _load_config(name="same", cache_dir=str(cache_dir))
    check_model_config_hash(loader)  # first load: stores the hash (and clears the empty cache)
    # Simulate the compile that just happened having filled the cache again.
    cache_dir.mkdir(parents=True)
    (cache_dir / "again.blob").write_bytes(b"compiled-again")

    reloaded = _load_config(name="same", cache_dir=str(cache_dir))
    _, recompiled = check_model_config_hash(reloaded)

    assert recompiled is False
    # The compiled cache must be left intact so the fast cache load is used.
    assert (cache_dir / "again.blob").exists()


def test_config_change_invalidates_cache_and_stores_new_hash(
    config_file: Path, tmp_path: Path
) -> None:
    """The core scenario: the entry changed (e.g. --runtime-config), the stored
    hash no longer matches, so the compiled cache is cleared and the new hash
    is stored."""
    cache_dir = _make_cache_dir(tmp_path, "changed")
    entry = {
        "model_name": "changed",
        "model_path": "/models/mock",
        "model_type": "llm",
        "engine": "ovgenai",
        "device": "CPU",
        "runtime_config": {},
        "cache_dir": str(cache_dir),
    }
    _write_entry(config_file, "changed", entry)
    check_model_config_hash(_load_config(name="changed", cache_dir=str(cache_dir)))

    # Operator edits the entry: new runtime config, hash still the old one.
    entry["runtime_config"] = {"OFFLOAD_RATIO": 0.05, "PERFORMANCE_HINT": "LATENCY"}
    _write_entry(config_file, "changed", entry)

    edited = _load_config(
        name="changed",
        runtime_config={"OFFLOAD_RATIO": 0.05, "PERFORMANCE_HINT": "LATENCY"},
        cache_dir=str(cache_dir),
    )
    current_hash, recompiled = check_model_config_hash(edited)

    assert recompiled is True
    assert not cache_dir.exists()  # stale compiled blobs are gone -> recompile
    assert _read_entry(config_file, "changed")[CONFIG_HASH_KEY] == current_hash

    # And a further load with the edited config is a cache hit again.
    _, recompiled_again = check_model_config_hash(
        _load_config(
            name="changed",
            runtime_config={"OFFLOAD_RATIO": 0.05, "PERFORMANCE_HINT": "LATENCY"},
            cache_dir=str(cache_dir),
        )
    )
    assert recompiled_again is False


def test_gate_ignores_models_not_in_config_file(config_file: Path, tmp_path: Path) -> None:
    """Raw POST /openarc/load without a config entry: nothing to track, and
    the cache is left alone so cache-less API users keep cache reuse."""
    cache_dir = _make_cache_dir(tmp_path, "ghost")
    loader = _load_config(name="ghost", cache_dir=str(cache_dir))
    current_hash, recompiled = check_model_config_hash(loader)
    assert recompiled is False
    assert current_hash
    assert json.loads(config_file.read_text(encoding="utf-8"))["models"] == {}
    assert cache_dir.exists()  # untouched


def test_gate_with_missing_cache_dir_is_a_noop(config_file: Path) -> None:
    _write_entry(
        config_file,
        "nocache",
        {"model_name": "nocache", "device": "CPU"},
    )
    loader = _load_config(name="nocache", cache_dir=None)
    current_hash, recompiled = check_model_config_hash(loader)
    assert recompiled is True  # first load of an entry without a hash
    assert current_hash
    assert _read_entry(config_file, "nocache")[CONFIG_HASH_KEY] == current_hash


# --- ModelRegistry integration ------------------------------------------------


def test_register_load_stores_hash_on_first_load(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, tmp_path: Path
) -> None:
    registry = _reg(monkeypatch, [])
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_entry(
        config_file,
        "reg-first",
        {
            "model_name": "reg-first",
            "model_path": str(model_path),
            "model_type": "llm",
            "engine": "ovgenai",
            "device": "CPU",
            "runtime_config": {},
        },
    )
    loader = _load_config(
        name="reg-first", model_path=str(model_path), cache_dir=None
    )

    async def _run():
        await registry.register_load(loader)
        async with registry._lock:
            record = next(r for r in registry._models.values())
            return record

    record = asyncio.run(_run())

    assert _read_entry(config_file, "reg-first")[CONFIG_HASH_KEY] == (
        compute_config_hash(record.load_config)
    )
    # The record carries the hash of the config it was compiled with ...
    assert record.load_config.config_hash == compute_config_hash(record.load_config)


def test_register_load_recompiles_on_stored_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, tmp_path: Path
) -> None:
    """Config edit while the model is unloaded: the next register_load clears
    the compiled cache and stores the new hash; a later OOM auto-reload from
    the record does NOT re-trigger the invalidation."""
    cache_dir = _make_cache_dir(tmp_path, "reg-re")
    model_path = tmp_path / "model"
    model_path.mkdir()
    entry = {
        "model_name": "reg-re",
        "model_path": str(model_path),
        "model_type": "llm",
        "engine": "ovgenai",
        "device": "CPU",
        "runtime_config": {},
        "cache_dir": str(cache_dir),
    }
    _write_entry(config_file, "reg-re", entry)

    invalidate_calls: list = []

    def counting_invalidate(cache_dir_value):  # type: ignore[override]
        invalidate_calls.append(cache_dir_value)
        return True

    monkeypatch.setattr(
        config_hash_module, "invalidate_compiled_model_cache", counting_invalidate
    )

    registry = _reg(monkeypatch, [])
    loader = _load_config(
        name="reg-re", model_path=str(model_path), cache_dir=str(cache_dir)
    )

    async def _run():
        await registry.register_load(loader)  # first load: stores hash, 1st invalidation
        await registry.register_unload("reg-re")
        await _await_unloaded(registry, "reg-re")

        # The operator edits --runtime-config in openarc_config.json.
        entry["runtime_config"] = {"OFFLOAD_RATIO": 0.1}
        _write_entry(config_file, "reg-re", entry)

        edited = _load_config(
            name="reg-re",
            model_path=str(model_path),
            runtime_config={"OFFLOAD_RATIO": 0.1},
            cache_dir=str(cache_dir),
        )
        await registry.register_load(edited)  # mismatch: 2nd invalidation

        # OOM auto-reload re-registers from the record's stored load config.
        await registry.reload_model("reg-re")

    asyncio.run(_run())

    # First load (no stored hash yet) + config-change load recompile; the
    # auto-reload from the record does not.
    assert invalidate_calls == [str(cache_dir), str(cache_dir)]
    assert (
        _read_entry(config_file, "reg-re")[CONFIG_HASH_KEY]
        == compute_config_hash(
            _load_config(
                name="reg-re",
                model_path=str(model_path),
                runtime_config={"OFFLOAD_RATIO": 0.1},
                cache_dir=str(cache_dir),
            )
        )
    )


def test_register_load_recompile_deletes_real_cache_dir(
    monkeypatch: pytest.MonkeyPatch, config_file: Path, tmp_path: Path
) -> None:
    """End-to-end on disk: a mismatch actually removes the stale blobs, and
    the unchanged reload after the config settled keeps its cache."""
    cache_dir = _make_cache_dir(tmp_path, "disk")
    model_path = tmp_path / "model"
    model_path.mkdir()
    entry = {
        "model_name": "disk",
        "model_path": str(model_path),
        "model_type": "llm",
        "engine": "ovgenai",
        "device": "CPU",
        "runtime_config": {},
        "cache_dir": str(cache_dir),
    }
    _write_entry(config_file, "disk", entry)
    registry = _reg(monkeypatch, [])
    loader = _load_config(name="disk", model_path=str(model_path), cache_dir=str(cache_dir))

    async def _run():
        await registry.register_load(loader)
        await registry.register_unload("disk")
        await _await_unloaded(registry, "disk")

        # Simulate a fresh compile having filled the cache, then a config edit.
        cache_dir.mkdir(parents=True)
        (cache_dir / "stale.blob").write_bytes(b"stale")
        entry["device"] = "GPU.0"
        _write_entry(config_file, "disk", entry)

        edited = _load_config(
            name="disk", model_path=str(model_path), cache_dir=str(cache_dir)
        )
        edited = edited.model_copy(update={"device": "GPU.0"})
        await registry.register_load(edited)

        # Next unchanged reload: the fake engine creates no cache artifacts,
        # so the gate must not re-trigger (hash matches now) and must cope
        # with the absent cache dir.
        await registry.register_unload("disk")
        await _await_unloaded(registry, "disk")
        again = _load_config(
            name="disk", model_path=str(model_path), cache_dir=str(cache_dir)
        ).model_copy(update={"device": "GPU.0"})
        await registry.register_load(again)

    asyncio.run(_run())

    assert not (cache_dir / "stale.blob").exists()  # stale blobs were removed
    stored = _read_entry(config_file, "disk")[CONFIG_HASH_KEY]
    assert stored == compute_config_hash(
        _load_config(
            name="disk", model_path=str(model_path), cache_dir=str(cache_dir)
        ).model_copy(update={"device": "GPU.0"})
    )


# --- openarc add: hash preservation on re-add ---------------------------------


def _add_cli(tmp_path: Path, name: str, model_dir: Path, *extra: str):
    from click.testing import CliRunner

    from src.cli import cli

    config_file = tmp_path / "openarc_config.json"
    args = [
        "add",
        "--model-name", name,
        "--model-path", str(model_dir),
        "--engine", "ovgenai",
        "--model-type", "llm",
        "--device", "CPU",
        *extra,
    ]
    return CliRunner().invoke(cli, args, env={"OPENARC_CONFIG_FILE": str(config_file)}), config_file


def test_add_preserves_hash_on_unchanged_readd(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<xml />", encoding="utf-8")
    (model_dir / "openvino_model.bin").write_bytes(b"bin")

    result, config_file = _add_cli(tmp_path, "readd", model_dir)
    assert result.exit_code == 0

    # Simulate a load having stored the hash.
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["models"]["readd"][CONFIG_HASH_KEY] = "sha256:old"
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Re-add with the identical configuration: the hash is preserved.
    result, _ = _add_cli(tmp_path, "readd", model_dir)
    assert result.exit_code == 0
    entry = json.loads(config_file.read_text(encoding="utf-8"))["models"]["readd"]
    assert entry[CONFIG_HASH_KEY] == "sha256:old"

    # Re-add with a CHANGED configuration: the hash is dropped, so the next
    # load recompiles.
    result, _ = _add_cli(tmp_path, "readd", model_dir, "--device", "GPU.0")
    assert result.exit_code == 0
    entry = json.loads(config_file.read_text(encoding="utf-8"))["models"]["readd"]
    assert CONFIG_HASH_KEY not in entry


def test_add_new_model_has_no_hash(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<xml />", encoding="utf-8")
    (model_dir / "openvino_model.bin").write_bytes(b"bin")

    result, config_file = _add_cli(tmp_path, "fresh", model_dir)
    assert result.exit_code == 0
    entry = json.loads(config_file.read_text(encoding="utf-8"))["models"]["fresh"]
    assert CONFIG_HASH_KEY not in entry
