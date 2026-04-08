"""MCP tool: synthesize speech via /v1/audio/speech and play on the server."""

from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from src.server.models.registration import ModelType

SPEAK_TOOL_NAME = "speak"
SPEAK_TOOL_TITLE = "Speak (OpenArc TTS)"
SPEAK_TOOL_DESCRIPTION = """
Use this tool to speak to the user.
""".strip()

_DEFAULT_L16_RATE = 24000


def _openarc_config_path() -> Path:
    # src/server/openarc_mcp/scripts/speak.py -> repo root
    return Path(__file__).resolve().parent.parent.parent.parent.parent / "openarc_config.json"


def _http_base_url(server_cfg: dict[str, Any]) -> str:
    host = str(server_cfg.get("host", "127.0.0.1"))
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    port = int(server_cfg.get("port", 8000))
    return f"http://{host}:{port}"


def _is_qwen3_tts(model_type: str) -> bool:
    try:
        mt = ModelType(model_type)
    except ValueError:
        return False
    return mt in (
        ModelType.QWEN3_TTS_CUSTOM_VOICE,
        ModelType.QWEN3_TTS_VOICE_DESIGN,
        ModelType.QWEN3_TTS_VOICE_CLONE,
    )


@dataclass(frozen=True)
class OpenArcMCPTTSConfig:
    """Loads and validates mcp.tts from openarc_config.json; builds /v1/audio/speech request template."""

    speech_url: str
    api_key: str
    body_template: dict[str, Any]

    @classmethod
    def load(
        cls,
        *,
        config_path: Path | None = None,
        api_key: str | None = None,
    ) -> OpenArcMCPTTSConfig:
        path = config_path or _openarc_config_path()
        if not path.is_file():
            raise FileNotFoundError(f"openarc_config.json not found: {path}")

        with open(path, encoding="utf-8") as f:
            config = json.load(f)

        mcp_cfg = config.get("mcp") or {}
        tts_cfg = mcp_cfg.get("tts")
        if not tts_cfg or not isinstance(tts_cfg, dict):
            raise ValueError('openarc_config.json: missing or invalid "mcp.tts" object')

        model_name = tts_cfg.get("model")
        if not model_name or not isinstance(model_name, str):
            raise ValueError('openarc_config.json: mcp.tts.model must be a non-empty string')

        models = config.get("models") or {}
        entry = models.get(model_name)
        if not entry:
            raise ValueError(
                f'openarc_config.json: mcp.tts.model "{model_name}" not found under "models"'
            )

        model_type = entry.get("model_type")
        if not model_type:
            raise ValueError(f'openarc_config.json: models["{model_name}"].model_type is required')

        key = api_key if api_key is not None else os.environ.get("OPENARC_API_KEY")
        if not key:
            raise ValueError("OPENARC_API_KEY must be set for MCP TTS (same key as /v1/audio/speech)")

        server_cfg = config.get("server") or {}
        base = _http_base_url(server_cfg)
        speech_url = f"{base.rstrip('/')}/v1/audio/speech"

        placeholder_input = ""

        if model_type == ModelType.KOKORO.value:
            kokoro_extra = copy.deepcopy(tts_cfg.get("kokoro") or {})
            if not isinstance(kokoro_extra, dict):
                raise ValueError("openarc_config.json: mcp.tts.kokoro must be an object")
            body: dict[str, Any] = {
                "model": model_name,
                "input": placeholder_input,
                "openarc_tts": {"kokoro": {**kokoro_extra, "input": placeholder_input}},
            }
        elif _is_qwen3_tts(model_type):
            q_extra = copy.deepcopy(tts_cfg.get("qwen3_tts") or {})
            if not isinstance(q_extra, dict):
                raise ValueError("openarc_config.json: mcp.tts.qwen3_tts must be an object")
            body = {
                "model": model_name,
                "input": placeholder_input,
                "openarc_tts": {"qwen3_tts": {**q_extra, "input": placeholder_input}},
            }
        else:
            raise ValueError(
                f'mcp.tts.model "{model_name}" has model_type "{model_type}"; '
                "expected kokoro or a qwen3_tts_* type"
            )

        return cls(speech_url=speech_url, api_key=key, body_template=body)


def _l16_rate_from_content_type(content_type: str) -> int:
    m = re.search(r"rate=(\d+)", content_type, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return _DEFAULT_L16_RATE


def _play_streaming_l16(response: httpx.Response, sample_rate: int) -> None:
    pending = bytearray()
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as out:
        for chunk in response.iter_bytes(chunk_size=8192):
            if not chunk:
                continue
            pending.extend(chunk)
            n_bytes = (len(pending) // 2) * 2
            if n_bytes == 0:
                continue
            raw = bytes(pending[:n_bytes])
            del pending[:n_bytes]
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if samples.size:
                out.write(samples.reshape(-1, 1))


def _play_wav_body(body: bytes) -> None:
    buf = io.BytesIO(body)
    audio_data, fs = sf.read(buf, dtype="float32")
    sd.play(audio_data, fs)
    sd.wait()


def _speak_sync(speech_url: str, api_key: str, body: dict) -> None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", speech_url, json=body, headers=headers) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "l16" in content_type.lower():
                sr = _l16_rate_from_content_type(content_type)
                _play_streaming_l16(response, sr)
                return

            chunks: list[bytes] = []
            for chunk in response.iter_bytes(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
            _play_wav_body(b"".join(chunks))


async def speak(
    input: Annotated[str, Field(description="Text to synthesize and play on the OpenArc server.")],
    ctx: Context,
) -> str:
    cfg: OpenArcMCPTTSConfig = ctx.request_context.lifespan_context
    body = copy.deepcopy(cfg.body_template)
    body["input"] = input
    oa = body.get("openarc_tts") or {}
    if "kokoro" in oa and oa["kokoro"] is not None:
        oa["kokoro"]["input"] = input
    if "qwen3_tts" in oa and oa["qwen3_tts"] is not None:
        oa["qwen3_tts"]["input"] = input

    await asyncio.to_thread(_speak_sync, cfg.speech_url, cfg.api_key, body)
    return ""


def register_openarc_tts_tool(mcp: FastMCP) -> None:
    """Register the speak tool on a FastMCP server (metadata lives with the implementation)."""
    mcp.add_tool(
        speak,
        name=SPEAK_TOOL_NAME,
        title=SPEAK_TOOL_TITLE,
        description=SPEAK_TOOL_DESCRIPTION,
    )
