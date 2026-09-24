"""Unit tests for context-window *enforcement* in the ov_genai engine.

The previous session made OpenArc *advertise* a model's context window in
``/v1/models``. These tests pin the follow-up: the resolved context window (an
explicit ``--context-window`` / ``config.yaml`` override, or the value
auto-discovered from the model's ``config.json``) must actually become the
compiled pipeline's **max content window** -- openvino.genai's
``SchedulerConfig.max_num_batched_tokens``, which is what bounds a running
sequence's KV-cache growth at inference time.

Needs ``openvino_genai`` (real pipeline types).
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest  # type: ignore[import]

# Skip the whole module where the real engine types are not importable, rather
# than erroring at collection (mirrors how the other ov_genai tests behave,
# but more gracefully for a minimal CI env).
pytest.importorskip("openvino_genai")

from openvino_genai import SchedulerConfig  # noqa: E402

import src.server.model_registry as registry_module  # noqa: E402
from src.server.model_registry import ModelRegistry  # noqa: E402
from src.engine.ov_genai.utils import (  # noqa: E402
    extract_scheduler_config_from_loader,
    generate_ov_scheduler_config,
)
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType  # noqa: E402
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import (  # noqa: E402
    SchedulerConfigSchema,
)


def _all_none_sched() -> SchedulerConfigSchema:
    """A scheduler_config block exactly like the all-blank template in config.yaml."""
    return SchedulerConfigSchema(
        max_num_batched_tokens=None,
        num_kv_blocks=None,
        cache_size=None,
        num_linear_attention_blocks=None,
        cache_interval_multiplier=None,
        dynamic_split_fuse=None,
        max_num_seqs=None,
        enable_prefix_caching=None,
        use_cache_eviction=None,
        use_sparse_attention=None,
    )


def _loader(**kwargs) -> ModelLoadConfig:
    base = dict(
        model_path="__fake__/ov",
        model_name="m",
        model_type=ModelType.LLM,
        engine=EngineType.OV_GENAI,
        device="CPU",
    )
    return ModelLoadConfig(**base, **kwargs)


def _enforced_max(loader: ModelLoadConfig) -> int:
    """The max content window openvino will *actually* enforce for this loader.

    When no scheduler_config is emitted, openvino falls back to its built-in
    (latency-oriented) default via extract_scheduler_config.
    """
    out = extract_scheduler_config_from_loader(loader)
    sched_config = out.get("scheduler_config", SchedulerConfig())
    return sched_config.max_num_batched_tokens


class TestWindowBecomesMaxContentWindow:
    def test_resolved_window_replaces_hidden_openvino_default(self):
        # Typical case: an all-blank scheduler block (as in config.yaml) plus a
        # resolved context_window -> the window becomes the max, replacing the
        # 256-token openvino default the operator never chose.
        lo = _loader(scheduler_config=_all_none_sched(), context_window=131072)
        assert _enforced_max(lo) == 131072

    def test_nothing_meaningful_set_leaves_openvino_default(self):
        # No scheduler values AND no context_window -> do not emit a
        # scheduler_config at all (so openvino uses its own default, uncapped),
        # instead of forcing the phantom 256 cap the old code did.
        lo = _loader(scheduler_config=_all_none_sched())
        assert "scheduler_config" not in extract_scheduler_config_from_loader(lo)

    def test_no_sched_block_and_no_window_does_not_emit(self):
        lo = _loader()
        assert "scheduler_config" not in extract_scheduler_config_from_loader(lo)

    def test_explicit_operator_value_wins_over_window(self):
        # An operator-set max_num_batched_tokens must always win over the window.
        lo = _loader(
            scheduler_config=SchedulerConfigSchema(max_num_batched_tokens=8192),
            context_window=131072,
        )
        assert _enforced_max(lo) == 8192

    def test_window_only_forcibly_emits_paged_scheduler(self):
        # No scheduler block at all, but a window -> force a (paged) scheduler and
        # set the window, so the value actually bites.
        lo = _loader(context_window=4096)
        out = extract_scheduler_config_from_loader(lo)
        assert "scheduler_config" in out
        assert out["scheduler_config"].max_num_batched_tokens == 4096

    def test_zero_and_negative_windows_are_absent(self):
        for bad in (0, -5, -1):
            lo = _loader(scheduler_config=_all_none_sched(), context_window=bad)
            assert "scheduler_config" not in extract_scheduler_config_from_loader(lo), bad
        lo = _loader(context_window=0)
        assert "scheduler_config" not in extract_scheduler_config_from_loader(lo)

    def test_window_and_kv_pool_coexist(self):
        # The window (per-sequence content cap) and the KV pool (num_kv_blocks) are
        # independent knobs; setting the window must not clobber an explicit pool.
        lo = _loader(
            scheduler_config=SchedulerConfigSchema(num_kv_blocks=4096),
            context_window=8192,
        )
        sched_config = extract_scheduler_config_from_loader(lo)["scheduler_config"]
        assert sched_config.max_num_batched_tokens == 8192
        assert sched_config.num_kv_blocks == 4096

    def test_sdpa_only_advertised_not_enforced(self, monkeypatch):
        # SDPA (non-paged) pipelines cannot take a scheduler_config, so the window
        # can only be *advertised* there; the compiled model's own baked
        # max_position_embeddings bounds the KV cache instead.
        lo = _loader(
            runtime_config={"ATTENTION_BACKEND": "SDPA"}, context_window=4096
        )
        out = extract_scheduler_config_from_loader(lo)
        assert out == {}


class TestConfigJsonDiscoveryReachesCompiledPipeline:
    def test_discovered_window_flows_through_register_load_to_scheduler(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End to end within a unit: config.json -> register_load resolves the value
        and writes it onto the loader the engine receives -> that same loader, run
        through extract_scheduler_config_from_loader, yields it as
        SchedulerConfig.max_num_batched_tokens.
        """
        model_dir = tmp_path / "ov_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"max_position_embeddings": 65536}), encoding="utf-8"
        )

        load_config = ModelLoadConfig(
            model_path=str(model_dir),
            model_name="discovered",
            model_type=ModelType.LLM,
            engine=EngineType.OV_GENAI,
            device="CPU",
            runtime_config={},
        )
        registry = ModelRegistry()

        seen = {"config": None}

        async def _noop_unload(*_args, **_kwargs):
            return None

        async def fake_create(config):
            seen["config"] = config
            return SimpleNamespace(unload_model=_noop_unload)

        monkeypatch.setattr(registry_module, "create_model_instance", fake_create)

        asyncio.run(registry.register_load(load_config))

        # The loader the engine actually received carries the DISCOVERED window...
        assert seen["config"].context_window == 65536
        # ...and folding it into the scheduler config makes it the max content window.
        assert _enforced_max(seen["config"]) == 65536
