"""Which request contract backs each config.yaml model-defaults block.

There is exactly one definition per config shape: the request contract in
``contract_*.py``. This module only maps block names to those contracts, so a
block can never drift from the contract it configures and adding a field to a
contract makes it configurable in config.yaml automatically.

Precedence, highest wins:

    request-time  >  config.yaml block  >  engine/pydantic default

A block is stored as a plain dict of only the keys the author actually wrote
(see ``ServerConfig.get_model_load_config``), and is merged under the request by
``src/server/utils/merge.py``. Because unset keys are simply absent, the
contract's own default applies when neither layer supplies a value.
"""
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from src.server.schemas.modeling.contract_kokoro import OV_KokoroGenConfig
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.modeling.contract_qwen3asr import OV_Qwen3ASRGenConfig
from src.server.schemas.modeling.contract_qwen3tts import (
    OV_Qwen3TTSCustomVoice,
    OV_Qwen3TTSGenConfig,
    OV_Qwen3TTSVoiceClone,
    OV_Qwen3TTSVoiceDesign,
)

# Block name -> (contract, model_type it applies to). A None model_type means
# the block is valid for any member of that family (see SHARED_TTS_BLOCK).
BLOCK_CONTRACTS: Dict[str, tuple[Type[BaseModel], Optional[str]]] = {
    "sampler_config": (OVGenAI_GenConfig, None),  # llm/vlm only, see SAMPLER_MODEL_TYPES
    "kokoro_config": (OV_KokoroGenConfig, "kokoro"),
    "qwen3_asr_config": (OV_Qwen3ASRGenConfig, "qwen3_asr"),
    "qwen3_tts_config": (OV_Qwen3TTSGenConfig, None),  # shared by all qwen3_tts_* modes
    "qwen3_tts_custom_voice_config": (OV_Qwen3TTSCustomVoice, "qwen3_tts_custom_voice"),
    "qwen3_tts_voice_design_config": (OV_Qwen3TTSVoiceDesign, "qwen3_tts_voice_design"),
    "qwen3_tts_voice_clone_config": (OV_Qwen3TTSVoiceClone, "qwen3_tts_voice_clone"),
}

# sampler_config applies only to llm/vlm.
SAMPLER_BLOCK = "sampler_config"
SAMPLER_MODEL_TYPES = {"llm", "vlm"}

# One block shared by every qwen3_tts_* mode.
SHARED_TTS_BLOCK = "qwen3_tts_config"
QWEN3_TTS_PREFIX = "qwen3_tts_"

# Fields that describe one specific request rather than a reusable model
# default. They may appear in a contract but must not be authored in config.yaml.
REQUEST_ONLY_FIELDS = frozenset(
    {
        "messages",
        "prompt",
        "input_ids",
        "tools",
        "tool_call_parser",
        "request_id",
        "chat_template_kwargs",
        "input",
        "audio_base64",
        "ref_audio_b64",
    }
)


def contract_for(block_name: str) -> Optional[Type[BaseModel]]:
    """Return the contract backing a block name, or None if unknown."""
    entry = BLOCK_CONTRACTS.get(block_name)
    return entry[0] if entry else None


def block_applies_to(block_name: str, model_type: str) -> bool:
    """Whether a config block may be used with a given model_type."""
    if block_name == SAMPLER_BLOCK:
        return model_type in SAMPLER_MODEL_TYPES
    if block_name == SHARED_TTS_BLOCK:
        return model_type.startswith(QWEN3_TTS_PREFIX)

    entry = BLOCK_CONTRACTS.get(block_name)
    if entry is None:
        return False
    required = entry[1]
    return required is not None and model_type == required


def validate_block(block_name: str, payload: Any, model_type: str) -> Dict[str, Any]:
    """Validate an authored block against its contract.

    Returns the block as a dict of only the authored keys.

    Raises:
        ValueError: If the block is unknown, does not apply to model_type, or
            contains a key the contract does not define.
    """
    contract = contract_for(block_name)
    if contract is None:
        raise ValueError(
            f"Unknown config block '{block_name}'. Valid blocks: "
            f"{', '.join(sorted(BLOCK_CONTRACTS))}"
        )
    if not block_applies_to(block_name, model_type):
        if block_name == SAMPLER_BLOCK:
            raise ValueError(
                f"sampler_config is only valid for {sorted(SAMPLER_MODEL_TYPES)} models, "
                f"but this model is model_type '{model_type}'"
            )
        entry = BLOCK_CONTRACTS[block_name]
        required = entry[1] or f"{QWEN3_TTS_PREFIX}*"
        raise ValueError(
            f"{block_name} requires model_type '{required}', "
            f"but this model is model_type '{model_type}'"
        )
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{block_name} must be a mapping, got {type(payload).__name__}")

    unknown = sorted(set(payload) - set(contract.model_fields))
    if unknown:
        raise ValueError(
            f"{block_name} has unknown key(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(contract.model_fields))}"
        )

    # Fields that describe one request rather than a reusable model default.
    request_only = sorted(set(payload) & REQUEST_ONLY_FIELDS)
    if request_only:
        raise ValueError(
            f"{block_name} cannot set request-only field(s): {', '.join(request_only)}"
        )

    # Coerce/validate values against the contract so a bad type or an invalid
    # enum member is caught here rather than at model load time.
    authored = {k: v for k, v in payload.items() if v is not None}
    try:
        validated = contract(**authored)
    except ValidationError as exc:
        raise ValueError(f"{block_name} has invalid values: {exc}") from exc

    # Re-read only the authored keys so unset fields stay absent.
    return {k: getattr(validated, k) for k in authored}


def model_block_name(model_type: str) -> Optional[str]:
    """Return the mode-specific block name for a model_type, if it has one."""
    for block_name, (_, required) in BLOCK_CONTRACTS.items():
        if required == model_type and block_name != SAMPLER_BLOCK:
            return block_name
    return None
