"""Utilities for normalising OpenAI-style chat payloads before tokenisation."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def _iter_text_fragments(node: Any) -> Iterable[str]:
    if node is None:
        return

    if isinstance(node, str):
        if node:
            yield node
        return

    if isinstance(node, list):
        for item in node:
            yield from _iter_text_fragments(item)
        return

    if isinstance(node, dict):
        node_type = node.get("type")

        if node_type == "text" and "text" in node:
            yield from _iter_text_fragments(node.get("text"))
            return

        if node_type in {"tool_result", "tool_response"}:
            for key in ("output", "content", "text", "result"):
                if key in node:
                    yield from _iter_text_fragments(node[key])
                    return

        for key in ("text", "content", "message", "output", "result"):
            if key in node:
                yield from _iter_text_fragments(node[key])
                return

        yield json.dumps(node, ensure_ascii=True, sort_keys=True)
        return

    yield str(node)


def flatten_message_content(content: Any) -> str:
    """Return a string representation suitable for chat templating."""

    return "\n".join(fragment for fragment in _iter_text_fragments(content))


def _arguments_as_mapping(arguments: Any) -> Any:
    """Qwen chat templates iterate ``arguments|items`` and need a mapping."""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
        return parsed if isinstance(parsed, dict) else arguments
    return arguments


def normalize_tool_calls_for_template(tool_calls: Any) -> Any:
    if not isinstance(tool_calls, list):
        return tool_calls
    normalized: List[Dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            normalized.append(call)
            continue
        fn = call.get("function")
        if isinstance(fn, dict) and "arguments" in fn:
            fn = {**fn, "arguments": _arguments_as_mapping(fn.get("arguments"))}
            normalized.append({**call, "function": fn})
        else:
            call = {**call, "arguments": _arguments_as_mapping(call.get("arguments"))}
            normalized.append(call)
    return normalized


def flatten_messages(messages: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if not messages:
        return []

    out: List[Dict[str, Any]] = []
    for message in messages:
        item = {**message, "content": flatten_message_content(message.get("content"))}
        if "tool_calls" in item:
            item["tool_calls"] = normalize_tool_calls_for_template(item.get("tool_calls"))
        out.append(item)
    return out

