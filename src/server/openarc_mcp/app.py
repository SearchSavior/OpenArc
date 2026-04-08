"""FastMCP Streamable HTTP app for OpenArc TTS (mounted under /openarc/tts)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from src.server.openarc_mcp.scripts.speak import OpenArcMCPTTSConfig, register_openarc_tts_tool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastMCP) -> AsyncIterator[OpenArcMCPTTSConfig]:
    cfg = OpenArcMCPTTSConfig.load()
    logger.info("MCP TTS: speech_url=%s model=%s", cfg.speech_url, cfg.body_template["model"])
    yield cfg


mcp = FastMCP("OpenArcTTS", lifespan=_lifespan, stateless_http=True)
register_openarc_tts_tool(mcp)
