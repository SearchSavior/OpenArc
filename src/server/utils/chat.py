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


def flatten_messages(messages: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if not messages:
        return []

    return [
        {**message, "content": flatten_message_content(message.get("content"))}
        for message in messages
    ]

