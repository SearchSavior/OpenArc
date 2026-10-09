"""Unit tests for the ov_genai engine's *operator-set* per-sequence cap.

``scheduler_config.max_num_batched_tokens`` (configured in ``config.yaml``).
These tests pin that the operator's own ``scheduler_config`` flows through to the
compiled ``openvino.genai.SchedulerConfig`` -- and that a PA-backend loader
always emits one, because prefix caching defaults on for PA (operator knobs
override the defaults; SDPA still emits nothing).

Needs ``openvino_genai`` (real pipeline types).
"""
import pytest  # type: ignore[import]

# Skip the whole module where the real engine types are not importable, rather
# than erroring at collection (mirrors how the other ov_genai tests behave,
# but more gracefully for a minimal CI env).
pytest.importorskip("openvino_genai")

from openvino_genai import SchedulerConfig  # noqa: E402

from src.engine.ov_genai.utils import (  # noqa: E402
    extract_scheduler_config_from_loader,
    generate_ov_scheduler_config,
)
from src.server.schemas.registration import EngineType, ModelLoadConfig, ModelType  # noqa: E402
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import (  # noqa: E402
    SchedulerConfigSchema,
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
    """The operator-set ``max_num_batched_tokens`` openvino will actually
    enforce for this loader (the engine's own cap).

    When no scheduler_config is emitted (operator left every field unset),
    openvino falls back to its built-in (latency-oriented) default.
    """
    out = extract_scheduler_config_from_loader(loader)
    sched_config = out.get("scheduler_config", SchedulerConfig())
    return sched_config.max_num_batched_tokens


class TestOperatorMaxNumBatchedTokens:
    def test_operator_value_becomes_compiled_max(self):
        # The normal case: the operator sets max_num_batched_tokens on an
        # otherwise-blank scheduler block -> it becomes the compiled max,
        # replacing openvino's hidden built-in default the operator never chose.
        lo = _loader(scheduler_config=SchedulerConfigSchema(max_num_batched_tokens=131072))
        out = extract_scheduler_config_from_loader(lo)
        assert "scheduler_config" in out
        assert out["scheduler_config"].max_num_batched_tokens == 131072

    def test_no_scheduler_block_emits_only_defaults(self):
        # No scheduler block at all -> the PA defaults are emitted (prefix
        # caching on, 8K prefill chunks); nothing operator-set.
        lo = _loader()
        sched_config = extract_scheduler_config_from_loader(lo)["scheduler_config"]
        assert sched_config.enable_prefix_caching is True
        assert sched_config.max_num_batched_tokens == 8192

    def test_empty_scheduler_block_emits_only_defaults(self):
        # A present-but-fully-unset scheduler block (all fields left None) is
        # the same as none: the PA defaults, nothing operator-set.
        lo = _loader(scheduler_config=SchedulerConfigSchema())
        sched_config = extract_scheduler_config_from_loader(lo)["scheduler_config"]
        assert sched_config.enable_prefix_caching is True
        assert sched_config.max_num_batched_tokens == 8192

    def test_operator_max_and_kv_pool_coexist(self):
        # max_num_batched_tokens (per-sequence content cap) and num_kv_blocks
        # (the total KV pool) are independent knobs; setting one must not clobber
        # the other.
        lo = _loader(
            scheduler_config=SchedulerConfigSchema(
                max_num_batched_tokens=8192, num_kv_blocks=4096
            )
        )
        sched_config = extract_scheduler_config_from_loader(lo)["scheduler_config"]
        assert sched_config.max_num_batched_tokens == 8192
        assert sched_config.num_kv_blocks == 4096

    def test_ignores_sdpa_backend(self):
        # SDPA (non-paged) pipelines cannot take a scheduler_config, so an
        # operator-set cap is rejected (an error is logged and nothing returned)
        # for that backend; the compiled model's own baked max_position_embeddings
        # bounds its KV cache instead.
        lo = _loader(
            runtime_config={"ATTENTION_BACKEND": "SDPA"},
            scheduler_config=SchedulerConfigSchema(max_num_batched_tokens=4096),
        )
        assert extract_scheduler_config_from_loader(lo) == {}


def test_context_window_does_not_drive_scheduler():
    """The context window is advertisement only: it never enters the compiled
    scheduler config. An opt-in context_window (a pinned int or "auto") with an
    all-unset scheduler block must leave every operator knob at openvino's
    built-in default -- proving the two are decoupled. (The PA prefix-caching
    default is still emitted.)
    """
    lo = _loader(context_window=131072)  # advertised value, no scheduler block
    sched_config = extract_scheduler_config_from_loader(lo)["scheduler_config"]
    assert sched_config.max_num_batched_tokens == 8192  # the tuned default, untouched
    assert sched_config.enable_prefix_caching is True
    # And the advertised value lands on the loader untouched, as-is.
    assert lo.context_window == 131072
