
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

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
    vlm_type: Optional[str] = Field(
        default=None,
        description="Deprecated legacy VLM token type. VLM tokens are resolved from config.json."
    )
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
        model_path.

        The cache is keyed by the model files and device, NOT by the
        runtime/scheduler configuration, so when this model's config entry
        changes (see config_hash) the server invalidates this directory before
        loading to force a recompile with the new settings.""")

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
    context_window: Optional[int] = Field(
        default=None,
        description="""
        Context window (in tokens) for this model. It is used in two places that
        must agree, both driven by this single value:

        * The compiled pipeline's **max content window** -- fed to the engine and
          written to openvino.genai's ``SchedulerConfig.max_num_batched_tokens``,
          which bounds a running sequence's KV-cache growth at inference time.
          An operator-set ``scheduler_config.max_num_batched_tokens`` must be the
          same value (or is left unset) so advertisement and enforcement match.
        * Advertised in ``/v1/models`` (both as the OpenAI-standard
          ``context_window`` field and as ``meta.n_ctx`` so goose can
          auto-compact against it).

        When set to a positive integer it overrides automatic discovery and is
        used as-is. When None (or <= 0) the value is discovered from the model's
        config.json -- searched at the top level first, and (because multimodal
        models nest their language config under ``text_config`` / ``language_config``
        / ``llm_config`` / ...) then inside those nested per-modality sections --
        using the first present key of
        max_position_embeddings / n_positions / seq_len / seq_length / n_ctx /
        sliding_window; that discovered value is likewise used as the
        compiled max content window.
        """,
    )
    config_hash: Optional[str] = Field(
        default=None,
        description="""
        Hash of this model's config entry as stored in openarc_config.json
        under the same key (see src.server.utils.config_hash).

        The server maintains it: at every load it hashes the configuration it
        is about to compile and compares it against the stored value. On a
        difference (a changed --runtime-config / --scheduler-config, a manual
        edit of openarc_config.json, a re-run of `openarc add`, ...) the
        model's compiled-model cache (cache_dir) is invalidated so the
        pipeline recompiles with the new settings, and the new hash is written
        back to the config file.

        Clients (openarc load) pass the stored value through unchanged; it
        never affects inference itself.""",
    )


class ModelUnloadConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model to unload")
