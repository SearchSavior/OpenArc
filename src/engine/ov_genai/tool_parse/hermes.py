"""Hermes JSON tool-call parser:

    <tool_call>
    {"name": NAME, "arguments": {...}}
    </tool_call>

Kept fully separate from the Qwen3.5 XML parser (qwen35.py). Only the
ReasoningSplitter and the partial-tag hold-back helper are shared (reasoning
splitting is not tool parsing).
"""
import json
import uuid
from typing import Any, Dict, List, Optional

from src.engine.ov_genai.tool_parse.qwen35 import ReasoningSplitter, _holdback_suffix

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


def _extract_hermes_tool_call_payloads(text: str) -> List[str]:
    payloads: List[str] = []
    cursor = 0

    while True:
        start = text.find(TOOL_OPEN, cursor)
        if start < 0:
            break

        payload_start = start + len(TOOL_OPEN)
        end = text.find(TOOL_CLOSE, payload_start)
        if end < 0:
            payload = text[payload_start:].strip()
            if payload:
                payloads.append(payload)
            break

        payload = text[payload_start:end].strip()
        if payload:
            payloads.append(payload)

        cursor = end + len(TOOL_CLOSE)

    return payloads


def _format_tool_call_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            return json.dumps(json.loads(arguments))
        except json.JSONDecodeError:
            return arguments
    return json.dumps(arguments)


def _payload_to_tool_call(payload: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not (isinstance(data, dict) and "name" in data and "arguments" in data):
        return None
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": str(data.get("name", "")),
            "arguments": _format_tool_call_arguments(data.get("arguments", {})),
        },
    }


def parse_hermes_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    tool_calls: List[Dict[str, Any]] = [
        tc
        for payload in _extract_hermes_tool_call_payloads(text)
        if (tc := _payload_to_tool_call(payload)) is not None
    ]
    return tool_calls if tool_calls else None


def parse_generation(
    text: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    enable_thinking: bool = True,
) -> tuple[str, str, Optional[List[Dict[str, Any]]]]:
    """Split model output into (reasoning, content, tool_calls).

    `tools` is accepted for dispatch symmetry with qwen35; hermes payloads are
    self-describing so it is unused.
    """
    reasoning, remainder = ReasoningSplitter(enabled="</think>" in text).feed(text)
    if remainder.startswith("<think>"):
        remainder = remainder[len("<think>") :]

    hermes = parse_hermes_tool_calls(remainder)
    if hermes:
        content = remainder
        for payload in _extract_hermes_tool_call_payloads(remainder):
            content = content.replace(f"{TOOL_OPEN}{payload}{TOOL_CLOSE}", "")
            content = content.replace(f"{TOOL_OPEN}\n{payload}\n{TOOL_CLOSE}", "")
        return reasoning, content.strip(), hermes

    return reasoning, remainder, None


THINK_OPEN = "<think>"


class HermesStreamParser:
    """Incremental hermes parser for streaming.

    Plain content streams live (with a hold-back on a partial <tool_call>
    prefix); JSON inside a call is buffered until </tool_call> (or stream end
    for unterminated payloads), then emitted as the two-fragment sequence:
    call-start-with-name, then arguments payload.

    Thinking is decided lazily from the stream start: hermes models that think
    emit a full <think>...</think> block, unlike qwen35 (whose chat template
    pre-opens the block). When enable_thinking is set, the first characters
    decide: a leading <think> enables reasoning splitting, anything else is
    plain content from the start.
    """

    def __init__(
        self,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: bool = True,
    ):
        self._undecided = enable_thinking
        self._pending = ""
        self._reasoning = ReasoningSplitter(enabled=False)
        self._buf = ""
        self._in_tool = False
        self._index = 0

    def feed(self, text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if self._undecided:
            self._pending += text
            stripped = self._pending.lstrip()
            if len(stripped) < len(THINK_OPEN) and THINK_OPEN.startswith(stripped):
                return []  # still could grow into a <think> tag
            self._undecided = False
            if stripped.startswith(THINK_OPEN):
                self._reasoning.in_reasoning = True
                text = stripped[len(THINK_OPEN):]
            else:
                text = self._pending
            self._pending = ""
        if not text:
            return out
        reasoning, content = self._reasoning.feed(text)
        if reasoning:
            out.append({"reasoning_content": reasoning})
        if content:
            self._buf += content
            out.extend(self._drain(final=False))
        return out

    def finish(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if self._undecided:
            self._undecided = False
            text = self._pending
            self._pending = ""
            if text:
                self._buf += text
        return out + self._drain(final=True)

    def _drain(self, final: bool) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        while self._buf:
            if not self._in_tool:
                idx = self._buf.find(TOOL_OPEN)
                if idx == -1:
                    hold = 0 if final else _holdback_suffix(self._buf, (TOOL_OPEN,))
                    safe, self._buf = self._buf[: len(self._buf) - hold], self._buf[len(self._buf) - hold:]
                    if safe:
                        out.append({"content": safe})
                    break
                if idx > 0:
                    out.append({"content": self._buf[:idx]})
                self._buf = self._buf[idx + len(TOOL_OPEN):]
                self._in_tool = True
            else:
                end = self._buf.find(TOOL_CLOSE)
                if end == -1:
                    if final:
                        self._emit_tool_calls(self._buf, out)
                        self._buf = ""
                    break
                payload = self._buf[:end]
                self._buf = self._buf[end + len(TOOL_CLOSE):]
                self._in_tool = False
                self._emit_tool_calls(payload, out)
        return out

    def _emit_tool_calls(self, payload: str, out: List[Dict[str, Any]]) -> None:
        tc = _payload_to_tool_call(payload.strip())
        if tc is None:
            return
        idx = self._index
        self._index += 1
        out.append(
            {
                "tool_calls": [
                    {
                        "index": idx,
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": "",
                        },
                    }
                ]
            }
        )
        out.append(
            {
                "tool_calls": [
                    {
                        "index": idx,
                        "function": {"arguments": tc["function"]["arguments"]},
                    }
                ]
            }
        )
