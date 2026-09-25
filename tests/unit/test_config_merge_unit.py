"""Precedence tests for the layered config merge.

    request-time  >  config.yaml block  >  engine/pydantic default

Config blocks are plain dicts on ModelRecord, keyed by block name, validated
against the request contract they configure (see config_blocks.BLOCK_CONTRACTS).
"""
from typing import Any, Dict

import pytest  # type: ignore[import]

from src.server.model_registry import ModelRecord
from src.server.schemas.modeling.config_blocks import validate_block
from src.server.schemas.modeling.contract_kokoro import OV_KokoroGenConfig
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.modeling.contract_qwen3asr import OV_Qwen3ASRGenConfig
from src.server.schemas.registration import EngineType, ModelType
from src.server.utils.merge import build_config, deep_merge, defaults_for_record, resolve_fields


def _record(
    model_type: ModelType, blocks: Dict[str, Dict[str, Any]] | None = None
) -> ModelRecord:
    return ModelRecord(
        model_name="m",
        model_type=model_type,
        engine=EngineType.OV_GENAI,
        model_config_blocks=blocks or {},
    )


# ---- deep_merge / resolve_fields ----


def test_deep_merge_recurses_into_nested_dicts() -> None:
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9, "z": 3}}) == {
        "a": {"x": 1, "y": 9, "z": 3}
    }


def test_deep_merge_replaces_lists_wholesale() -> None:
    assert deep_merge({"a": [1, 2, 3]}, {"a": [9]}) == {"a": [9]}


def test_resolve_fields_request_none_does_not_clobber_default() -> None:
    assert resolve_fields(request={"temperature": None}, defaults={"temperature": 0.7}) == {
        "temperature": 0.7
    }


def test_resolve_fields_request_wins_over_default() -> None:
    assert resolve_fields(request={"temperature": 0.1}, defaults={"temperature": 0.7}) == {
        "temperature": 0.1
    }


def test_resolve_fields_drops_none_defaults() -> None:
    assert resolve_fields(request={}, defaults={"top_k": None, "top_p": 0.9}) == {"top_p": 0.9}


# ---- three-layer precedence ----


def test_yaml_default_applies_when_request_omits_field() -> None:
    config = build_config(
        OVGenAI_GenConfig,
        request={"temperature": 0.1},
        defaults={"temperature": 0.7, "top_k": 40},
        messages=[],
    )
    assert config.temperature == 0.1  # request wins
    assert config.top_k == 40  # yaml fills the rest


def test_engine_default_applies_when_neither_layer_sets_field() -> None:
    config = build_config(OVGenAI_GenConfig, request={}, defaults={}, messages=[])
    assert config.top_k == 50  # OVGenAI_GenConfig default


def test_fields_set_contains_only_provided_fields() -> None:
    """Purely-defaulted fields must stay out of model_fields_set.

    routes/openai.py gates its top-level overrides on this, so a field no layer
    provided must not look explicitly set.
    """
    config = build_config(
        OVGenAI_GenConfig, request={"temperature": 0.3}, defaults={}, messages=[]
    )
    assert "temperature" in config.model_fields_set
    assert "seed" not in config.model_fields_set


def test_build_config_ignores_keys_not_on_contract() -> None:
    config = build_config(
        OVGenAI_GenConfig,
        request={"not_a_field": 1},
        defaults={"also_not_a_field": 2},
        messages=[],
    )
    assert not hasattr(config, "not_a_field")


# ---- block validation against contracts ----


def test_validate_block_accepts_known_keys() -> None:
    assert validate_block("sampler_config", {"temperature": 0.7}, "llm") == {"temperature": 0.7}


def test_validate_block_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        validate_block("sampler_config", {"temperatur": 0.7}, "llm")


def test_validate_block_drops_none_values() -> None:
    assert validate_block("sampler_config", {"temperature": None, "top_k": 5}, "llm") == {
        "top_k": 5
    }


def test_validate_block_rejects_model_type_mismatch() -> None:
    with pytest.raises(ValueError, match="requires model_type"):
        validate_block("kokoro_config", {"voice": "af_sarah"}, "llm")


def test_validate_block_rejects_sampler_for_non_llm() -> None:
    with pytest.raises(ValueError, match="sampler_config is only valid"):
        validate_block("sampler_config", {"temperature": 0.7}, "kokoro")


def test_validate_block_rejects_unknown_block() -> None:
    with pytest.raises(ValueError, match="Unknown config block"):
        validate_block("bogus_config", {}, "llm")


def test_validate_block_shared_tts_applies_to_every_mode() -> None:
    for mode in (
        "qwen3_tts_custom_voice",
        "qwen3_tts_voice_design",
        "qwen3_tts_voice_clone",
    ):
        assert validate_block("qwen3_tts_config", {"top_k": 5}, mode) == {"top_k": 5}


def test_validate_block_mode_block_rejected_for_other_mode() -> None:
    with pytest.raises(ValueError, match="requires model_type"):
        validate_block("qwen3_tts_voice_clone_config", {"ref_text": "x"}, "qwen3_tts_voice_design")


def test_validate_block_rejects_request_only_field() -> None:
    # `messages` is not a model-level default; the contract defines it but a
    # config author should not be able to set it.
    with pytest.raises(ValueError, match="request-only field"):
        validate_block("sampler_config", {"messages": [{"role": "user"}]}, "llm")


def test_validate_block_accepts_chat_template_kwargs_default() -> None:
    # Thinking behavior is a reusable model default, so chat_template_kwargs
    # is a legitimate sampler_config block key.
    assert validate_block(
        "sampler_config",
        {"chat_template_kwargs": {"enable_thinking": False}},
        "llm",
    ) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_validate_block_rejects_load_config_field() -> None:
    # tool_call_parser is selected at load time; authoring it in a block should
    # point the author at load_config.
    with pytest.raises(ValueError, match="load-time field"):
        validate_block("sampler_config", {"tool_call_parser": "qwen35"}, "llm")


# ---- defaults_for_record ----


def test_defaults_for_record_reads_sampler_for_llm() -> None:
    record = _record(ModelType.LLM, blocks={"sampler_config": {"temperature": 0.7}})
    assert defaults_for_record(record) == {"temperature": 0.7}


def test_defaults_for_record_reads_block_for_kokoro() -> None:
    record = _record(ModelType.KOKORO, blocks={"kokoro_config": {"voice": "af_sarah"}})
    assert defaults_for_record(record) == {"voice": "af_sarah"}


def test_defaults_for_record_ignores_mismatched_block() -> None:
    record = _record(ModelType.LLM, blocks={"kokoro_config": {"voice": "af_sarah"}})
    assert defaults_for_record(record) == {}


def test_defaults_for_record_merges_shared_and_mode_tts_blocks() -> None:
    record = _record(
        ModelType.QWEN3_TTS_VOICE_CLONE,
        blocks={
            "qwen3_tts_config": {"top_k": 5, "temperature": 0.9},
            "qwen3_tts_voice_clone_config": {"ref_text": "hi"},
        },
    )
    assert defaults_for_record(record) == {"top_k": 5, "temperature": 0.9, "ref_text": "hi"}


def test_mode_block_wins_over_shared_tts_block() -> None:
    record = _record(
        ModelType.QWEN3_TTS_VOICE_CLONE,
        blocks={
            "qwen3_tts_config": {"top_k": 5},
            "qwen3_tts_voice_clone_config": {"top_k": 99},
        },
    )
    assert defaults_for_record(record)["top_k"] == 99


def test_defaults_for_record_handles_empty_record() -> None:
    assert defaults_for_record(_record(ModelType.LLM)) == {}


# ---- end-to-end per-contract precedence ----


def test_kokoro_yaml_default_yields_to_request() -> None:
    record = _record(ModelType.KOKORO, blocks={"kokoro_config": {"voice": "af_sarah", "speed": 1.0}})
    config = build_config(
        OV_KokoroGenConfig,
        request={"speed": 2.0},
        defaults=defaults_for_record(record),
        input="hi",
    )
    assert config.speed == 2.0  # request wins
    assert config.voice.value == "af_sarah"  # yaml default survives
    assert config.character_count_chunk == 400  # engine default


def test_asr_yaml_default_yields_to_request() -> None:
    record = _record(
        ModelType.QWEN3_ASR,
        blocks={"qwen3_asr_config": {"language": "English", "max_chunk_sec": 25.0}},
    )
    config = build_config(
        OV_Qwen3ASRGenConfig,
        request={"audio_base64": "AAA=", "language": "Chinese"},
        defaults=defaults_for_record(record),
    )
    assert config.language == "Chinese"  # request wins
    assert config.max_chunk_sec == 25.0  # yaml default
    assert config.max_tokens == 1024  # engine default
