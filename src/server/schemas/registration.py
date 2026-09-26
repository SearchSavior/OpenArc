
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import SchedulerConfigSchema


class ModelStatus(str, Enum):
    """loading status.

    Options:
    - LOADING: Model is currently being loaded in the background
    - LOADED: Model has been successfully loaded and is ready for inference
    - FAILED: Model loading failed
    """
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


class ModelType(str, Enum):
    """
    Internal routing to the correct inference pipeline.

    Options:
    - llm: Text-to-text LLM models
    - vlm: Image-to-text VLM models
    - whisper: Whisper ASR models
    - qwen3_asr: Qwen3 ASR models
    - kokoro: Kokoro TTS models
    - qwen3_tts_custom_voice: Qwen3-TTS with predefined speaker
    - qwen3_tts_voice_design: Qwen3-TTS with free-form voice description
    - qwen3_tts_voice_clone: Qwen3-TTS cloning a reference audio
    - emb: Text-to-vector models
    - rerank: Reranker models"""

    LLM = "llm"
    VLM = "vlm"
    WHISPER = "whisper"
    QWEN3_ASR = "qwen3_asr"
    KOKORO = "kokoro"
    QWEN3_TTS_CUSTOM_VOICE = "qwen3_tts_custom_voice"
    QWEN3_TTS_VOICE_DESIGN = "qwen3_tts_voice_design"
    QWEN3_TTS_VOICE_CLONE = "qwen3_tts_voice_clone"
    EMB = "emb"
    RERANK = "rerank"


class EngineType(str, Enum):
    """Engine used to load the model.

    Options:
    - optimum: Optimum-Intel engine
    - ovgenai: OpenVINO GenAI engine"""

    OV_OPTIMUM = "optimum"
    OV_GENAI = "ovgenai"
    OPENVINO = "openvino"


class ToolCallParser(str, Enum):
    """Tool-call output format the model was trained to emit, selected at load time.

    Options:
    - qwen35: Qwen3.5 XML format (<tool_call><function=NAME><parameter=KEY>...)
    - hermes: Hermes JSON format (<tool_call>{"name": ..., "arguments": {...}}</tool_call>)
    - gemma4: Gemma 4 call syntax (<|tool_call>call:NAME{KEY:VALUE, ...}<tool_call|>)
    - museglimmer: Muse-Glimmer Harmony 'to=' channels with atem XML payloads
      (<|start|>assistant to=NAME<|message|><atem:function_calls>...)"""

    HERMES_PARSER = "hermes"
    QWEN35_PARSER = "qwen35"
    GEMMA4_PARSER = "gemma4"
    MUSEGLIMMER_PARSER = "museglimmer"


class ModelLoadConfig(BaseModel):
    # Strict: an unrecognized key is a config error, not something to silently
    # drop. Typos in hand-written config.yaml fail loudly at load time.
    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(
        description="""
        Top level path to directory containing OpenVINO IR converted model.

        OpenArc does not support runtime conversion and cannot pull from HF.""")
    model_name: str = Field(
        ...,
        description="""
        - Public facing name of the loaded model attached to a private model_id
        - Calling /v1/models will report loaded models by model_name.
        """
    )
    model_type: ModelType = Field(...)
    engine: EngineType = Field(...)
    device: str = Field(
        ...,
        description="""
        Device used to load the model.
        """
    )
    runtime_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional OpenVINO runtime properties.")
    cache_dir: Optional[str] = Field(
        default=None,
        description="""
        Optional directory for the OpenVINO model cache (CACHE_DIR property).

        When set, compiled model blobs are cached here so subsequent loads of
        this model skip recompilation. Relative paths are resolved against the
        config file's directory when the model is loaded, the same as
        model_path.""")

    draft_model_path: Optional[str] = Field(
        default=None,
        description="Path to draft model for speculative decoding. Enables 1.3-1.4x speedup."
    )
    draft_device: Optional[str] = Field(
        default="CPU",
        description="Device for draft model (CPU, GPU, GPU.0, GPU.1)"
    )
    num_assistant_tokens: Optional[int] = Field(
        default=None,
        description="Default num_assistant_tokens for speculative decoding with this model"
    )
    assistant_confidence_threshold: Optional[float] = Field(
        default=None,
        description="Default assistant_confidence_threshold for speculative decoding with this model"
    )
    scheduler_config: Optional[SchedulerConfigSchema] = Field(
        default=None,
        description="Optional OpenVINO scheduler properties.",
    )
    tool_call_parser: Optional[ToolCallParser] = Field(
        default=None,
        description="""
        Tool-call parser for this model, selected at load time (llm/vlm only).

        When unset, /chat/completions requests containing tools are rejected
        with 400.""",
    )
    worker_line_limit: Optional[int] = Field(
        default=None,
        ge=65536,
        description="""
        Maximum size, in bytes, of one line on the IPC pipe between the server
        and this model's inference worker subprocess (ovgenai models only;
        ignored by in-process engines).

        A request whose JSON line exceeds the limit fails with a clear error
        (the worker reports FATAL and the supervisor respawns it). When unset,
        the protocol default of 256 MiB applies, which covers arbitrarily long
        text conversations, dozens of images, and hours of audio. Lower it to
        bound IPC memory on small machines or to reject oversized payloads;
        raise it for very large single requests.

        This is an IPC setting, not a compilation setting: changing it never
        invalidates the compiled-model cache (it is excluded from config_hash).
        """,
    )

    # --- Model-level request defaults, authored in config.yaml ---
    # Each entry is a plain dict of only the keys the author wrote, keyed by
    # block name (e.g. 'sampler_config', 'kokoro_config'). Validated against the
    # matching request contract by validate_config_blocks().
    # These seed per-request configs; anything the client sends wins.
    # Precedence: request > config.yaml block > contract default.
    model_config_blocks: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Model-level request defaults keyed by block name. Each block is "
            "validated against the request contract it configures."
        ),
    )

    def validate_config_blocks(self) -> None:
        """Validate every supplied config block against its contract.

        Checks that each block name is known, applies to this model's
        model_type, and contains no key the contract does not define. Unknown
        keys are an error so a typo fails loudly instead of being ignored.

        Also normalizes each block, dropping unset (None) values so that
        "not authored" stays distinct from "authored as the contract default".

        Raises:
            ValueError: If any block is unknown, mismatched, or has bad keys.
        """
        from src.server.schemas.modeling.config_blocks import validate_block

        model_type = self.model_type.value
        validated: Dict[str, Dict[str, Any]] = {}

        for block_name, payload in (self.model_config_blocks or {}).items():
            try:
                resolved = validate_block(block_name, payload, model_type)
            except ValueError as exc:
                raise ValueError(f"'{self.model_name}': {exc}") from exc
            if resolved:
                validated[block_name] = resolved

        self.model_config_blocks = validated

class ModelUnloadConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model to unload")
