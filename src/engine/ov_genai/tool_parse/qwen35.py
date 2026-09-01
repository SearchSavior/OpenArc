"""Qwen3.5 XML tool-call parser built on openvino_genai IncrementalParser.

Tool-call block boundaries are detected by TOKEN ID, not text matching:
  - a block opens only in a delta whose delta_tokens contains TOOL_OPEN_ID
  - a block closes only in a delta whose delta_tokens contains TOOL_CLOSE_ID
  - the boundary tags are single special=False tokens in the Qwen3.5 vocab, so
    their text always arrives in the same delta as their ID (even with
    skip_special_tokens=True, which only strips true specials like <|im_end|>)
  - the tag text is only used to slice the captured payload, never to decide
    that a boundary exists

Two consumers:
  Qwen35ToolCallStreamer - StreamerBase subclass used by the engine for
                           streaming tool requests; does its own incremental
                           decode, matches boundary tags by token ID, and
                           enqueues {"chat_delta": [...]} dicts. Deliberately
                           avoids TextParserStreamer, which deadlocks when
                           generate runs on a worker thread under asyncio.
  Qwen35StreamParser     - text-delta facade over ChunkStreamer output; feeds
                           ReasoningSplitter + QwenXMLToolParser, synthesizing
                           boundary token IDs from the tag text.

Parsed OpenAI streaming fragments:
    {"index": i, "id": ..., "type": "function",
     "function": {"name": "", "arguments": ""}}          # call start
    {"index": i, "function": {"name": NAME}}             # name known
    {"index": i, "function": {"arguments": FRAGMENT}}    # argument deltas

Concatenating all `arguments` fragments for an index yields complete JSON.
"""
import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from openvino_genai import IncrementalParser, StreamerBase, StreamingStatus

logger = logging.getLogger(__name__)

# Qwen3.5 XML format. Boundary tags are single special=False tokens; the inner
# tags (<function=, <parameter=...) are multi-token ordinary text and are
# matched at the text level inside a block.
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
FUNC_OPEN = "<function="
FUNC_CLOSE = "</function>"
PARAM_OPEN = "<parameter="
PARAM_CLOSE = "</parameter>"
GT = ">"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Token IDs of the boundary tags (validated identical across Qwen3.5 exports).
TOOL_OPEN_ID = 248058
TOOL_CLOSE_ID = 248059
BOUNDARY_IDS = (TOOL_OPEN_ID, TOOL_CLOSE_ID)

# Parser states
OUTSIDE = "outside"        # no open block: text is content (or ramble after a call)
HEAD = "head"              # awaiting "<function="
FUNC_NAME = "func_name"    # awaiting ">" that ends the function name
BODY = "body"              # inside <function>: params or </function>
PARAM_NAME = "param_name"  # awaiting ">" that ends the parameter name
PARAM = "param"            # inside a parameter value
TAIL = "tail"              # after </function>, awaiting the close boundary


def _holdback_suffix(text: str, tags) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of a tag.

    Anything before that suffix is safe to process now; the suffix must wait
    for more deltas (it could grow into a full tag).
    """
    max_hold = 0
    upper = max(len(t) for t in tags) - 1
    for n in range(1, min(len(text), upper) + 1):
        suffix = text[-n:]
        if any(t != suffix and t.startswith(suffix) for t in tags):
            max_hold = n
    return max_hold


def _is_partial(text: str, tags) -> bool:
    """True if `text` is a proper prefix of one of `tags`."""
    return any(t != text and t.startswith(text) for t in tags)


def _json_escape_fragment(s: str) -> str:
    """Escape a raw string fragment for embedding inside a JSON string."""
    out = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


class ReasoningSplitter:
    """Splits the leading thinking block from regular content.

    Qwen3.5 with enable_thinking=True emits reasoning immediately (no opening
    think tag) and terminates it with the close tag. Everything before the
    close tag is reasoning; everything after is content.
    """

    def __init__(self, close_tag: str = THINK_CLOSE, enabled: bool = True):
        self._close_tag = close_tag
        self._buf = ""
        # Only treat the stream as reasoning when the prompt actually opened a
        # think block (chat template appends it when enable_thinking=True; with
        # False it pre-closes an empty block and the stream is pure content).
        self.in_reasoning = enabled

    def feed(self, text: str) -> Tuple[str, str]:
        """Returns (reasoning_delta, content_delta)."""
        self._buf += text
        reasoning_out, content_out = "", ""
        while self._buf:
            if not self.in_reasoning:
                content_out += self._buf
                self._buf = ""
                break
            idx = self._buf.find(self._close_tag)
            if idx != -1:
                reasoning_out += self._buf[:idx]
                self._buf = self._buf[idx + len(self._close_tag):]
                self.in_reasoning = False
                continue
            hold = _holdback_suffix(self._buf, (self._close_tag,))
            reasoning_out += self._buf[: len(self._buf) - hold]
            self._buf = self._buf[len(self._buf) - hold:]
            break
        return reasoning_out, content_out


class _TextTagSynthesizer:
    """Stateful text adapter upholding the parser's boundary invariant.

    Buffers text and emits (chunk, delta_tokens) pairs such that every
    complete boundary tag arrives as its own chunk carrying its token ID, and
    no chunk ends with a partial boundary tag. Used by text-only feeders
    (ChunkStreamer path, parse_generation) where no real token IDs exist.
    """

    def __init__(self):
        self._buf = ""

    def feed(self, text: str) -> List[Tuple[str, Optional[List[int]]]]:
        self._buf += text
        out: List[Tuple[str, Optional[List[int]]]] = []
        while self._buf:
            events = []
            for tag, tag_id in ((TOOL_OPEN, TOOL_OPEN_ID), (TOOL_CLOSE, TOOL_CLOSE_ID)):
                pos = self._buf.find(tag)
                if pos != -1:
                    events.append((pos, tag, tag_id))
            if events:
                pos, tag, tag_id = min(events)
                if pos > 0:
                    out.append((self._buf[:pos], None))
                    self._buf = self._buf[pos:]
                out.append((tag, [tag_id]))
                self._buf = self._buf[len(tag):]
                continue
            hold = _holdback_suffix(self._buf, (TOOL_OPEN, TOOL_CLOSE))
            keep = len(self._buf) - hold
            if keep > 0:
                out.append((self._buf[:keep], None))
                self._buf = self._buf[keep:]
            break
        return out

    def flush(self) -> str:
        """Release the held tail at end of stream (it never became a tag)."""
        chunk, self._buf = self._buf, ""
        return chunk


class QwenXMLToolParser(IncrementalParser):
    """IncrementalParser for Qwen3.5 XML tool calls, bound by token IDs.

    Boundary detection is token-ID only:
      - A block opens only in a delta whose delta_tokens contains TOOL_OPEN_ID.
      - A block closes only in a delta whose delta_tokens contains TOOL_CLOSE_ID.
      - Boundary tag text is sliced out of the accumulated text; text without
        token info drains as ordinary content.

    Parameter values are emitted as JSON strings (no schema-driven type
    coercion). `on_fragment` receives OpenAI streaming fragments as they are
    produced. With
    stop_after_tool_call=True the parser requests StreamingStatus.TOOL_CALL_STOP
    right after each complete call; by default (False) sequential parallel
    calls are allowed and generation is only stopped when real text follows a
    completed call.
    """

    def __init__(
        self,
        on_fragment: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_after_tool_call: bool = False,
    ):
        super().__init__()
        self.on_fragment = on_fragment
        self._stop_after = stop_after_tool_call
        self.reset()

    def reset(self) -> None:
        self._state = OUTSIDE
        self._pending = ""
        self._cur: Optional[Dict[str, Any]] = None   # call under construction
        self._cur_index = -1
        self._calls_seen = 0                          # calls started (open boundary seen)
        self._calls: List[Dict[str, Any]] = []        # completed calls, OpenAI shape
        self._param_name = ""
        self._param_raw = ""
        self._param_index = 0
        self._value_started = False
        self._pending_newline = False
        self._skip_rest = False
        self._stopped = False
        self.errors: List[str] = []
        self.status = StreamingStatus.RUNNING
        self.set_status(StreamingStatus.RUNNING)

    @property
    def completed_calls(self) -> List[Dict[str, Any]]:
        """All completed calls, in final OpenAI non-streaming shape."""
        return list(self._calls)

    def finalize(self) -> None:
        """Call at end of generation: force-close an unterminated call and
        surface malformed structures in `errors`."""
        if self._cur is not None:
            self.errors.append("generation ended with unterminated tool call")
            self._finish_call({})
        elif self._state not in (OUTSIDE,):
            self.errors.append(f"generation ended in state {self._state}")

    # -- IncrementalParser API ---------------------------------------------------

    def parse(self, msg: dict, delta_text: str, delta_tokens=None) -> str:
        """Consume one decoded delta. Returns content text for this delta.

        Mutates `msg` with completed calls under msg["tool_calls"][str(index)]
        and emits streaming fragments through `on_fragment`.
        """
        if self._stopped:
            return ""
        self._pending += delta_text
        remaining: Dict[int, int] = {}
        if delta_tokens:
            for token in delta_tokens:
                tid = int(token)
                if tid in BOUNDARY_IDS:
                    remaining[tid] = remaining.get(tid, 0) + 1

        out: List[str] = []
        while True:
            # Earliest boundary event: ID still available AND tag text in pending
            events = []
            for tag_id, tag in ((TOOL_OPEN_ID, TOOL_OPEN), (TOOL_CLOSE_ID, TOOL_CLOSE)):
                if remaining.get(tag_id):
                    pos = self._pending.find(tag)
                    if pos != -1:
                        events.append((pos, tag_id, tag))
            if not events:
                self._machine(out, final=False)
                break
            pos, tag_id, tag = min(events)
            segment = self._pending[:pos]
            remainder = self._pending[pos + len(tag):]
            # The machine must see exactly the text before the boundary;
            # final=True guarantees it drains the segment completely.
            self._pending = segment
            if segment:
                self._machine(out, final=True)
            self._pending = remainder
            remaining[tag_id] -= 1
            self._apply_boundary(tag_id, msg)
            if self._stop_after and self.status == StreamingStatus.TOOL_CALL_STOP:
                self._pending = ""
                break
        return "".join(out)

    # -- state machine ------------------------------------------------------------

    def _machine(self, out: List[str], final: bool) -> None:
        """Advance the machine over `self._pending`.

        final=True means the segment ends at a boundary: nothing may be held
        back waiting for more text.
        """
        while True:
            if self._skip_rest:
                self._pending = ""
                return
            if self._state == OUTSIDE:
                if not self._pending:
                    return
                text, self._pending = self._pending, ""
                self._emit_outside(out, text)
                return
            if self._state == TAIL:
                if self._pending.strip():
                    self.errors.append(
                        f"unexpected text after </function>: {self._pending[:20]!r}"
                    )
                self._pending = ""
                return
            if self._state == HEAD:
                stripped = self._pending.lstrip()
                self._pending = stripped
                if stripped.startswith(FUNC_OPEN):
                    self._pending = stripped[len(FUNC_OPEN):]
                    self._state = FUNC_NAME
                    continue
                if not stripped:
                    return
                if not final and _is_partial(stripped, (FUNC_OPEN,)):
                    return
                self.errors.append(
                    f"expected '<function=' after tool open, got {stripped[:20]!r}"
                )
                self._skip_rest = True
                self._pending = ""
                return
            if self._state == FUNC_NAME:
                gt = self._pending.find(GT)
                if gt == -1:
                    if "<" in self._pending or final:
                        self.errors.append(
                            f"malformed function name: {self._pending[:20]!r}"
                        )
                        self._skip_rest = True
                        self._pending = ""
                    return
                name = self._pending[:gt].strip()
                self._pending = self._pending[gt + 1:]
                if self._cur is not None:
                    self._cur["function"]["name"] = name
                    self._frag({"index": self._cur_index, "function": {"name": name}})
                self._state = BODY
                continue
            if self._state == BODY:
                stripped = self._pending.lstrip()
                self._pending = stripped
                if stripped.startswith(PARAM_OPEN):
                    self._pending = stripped[len(PARAM_OPEN):]
                    self._state = PARAM_NAME
                    continue
                if stripped.startswith(FUNC_CLOSE):
                    self._pending = stripped[len(FUNC_CLOSE):]
                    self._state = TAIL
                    continue
                if not stripped:
                    return
                if not final and _is_partial(stripped, (PARAM_OPEN, FUNC_CLOSE)):
                    return
                self.errors.append(
                    f"unexpected text inside <function>: {stripped[:20]!r}"
                )
                self._pending = stripped[1:]
                continue
            if self._state == PARAM_NAME:
                gt = self._pending.find(GT)
                if gt == -1:
                    if "<" in self._pending or final:
                        self.errors.append(
                            f"malformed parameter name: {self._pending[:20]!r}"
                        )
                        self._skip_rest = True
                        self._pending = ""
                    return
                self._param_name = self._pending[:gt].strip()
                self._pending = self._pending[gt + 1:]
                self._param_raw = ""
                self._value_started = False
                self._pending_newline = False
                self._state = PARAM
                sep = "{" if self._param_index == 0 else ", "
                self._param_index += 1
                self._frag({
                    "index": self._cur_index,
                    "function": {"arguments": f"{sep}{json.dumps(self._param_name)}: "},
                })
                self._frag({"index": self._cur_index, "function": {"arguments": '"'}})
                continue
            if self._state == PARAM:
                idx = self._pending.find(PARAM_CLOSE)
                if idx == -1:
                    if final:
                        value, self._pending = self._pending, ""
                        self._consume_value(value, final=True)
                        self._state = BODY
                        continue
                    hold = _holdback_suffix(self._pending, (PARAM_CLOSE,))
                    safe = self._pending[: len(self._pending) - hold]
                    self._pending = self._pending[len(self._pending) - hold:]
                    if safe:
                        self._consume_value(safe, final=False)
                    return
                value, self._pending = (
                    self._pending[:idx],
                    self._pending[idx + len(PARAM_CLOSE):],
                )
                self._consume_value(value, final=True)
                self._state = BODY
                continue
            return

    def _apply_boundary(self, tag_id: int, msg: Optional[dict] = None) -> None:
        if tag_id == TOOL_OPEN_ID:
            if self._cur is not None:
                self.errors.append("new tool call opened before previous closed")
                self._finish_call(msg if msg is not None else {})
            self._start_call()
        else:
            if self._state == TAIL:
                self._finish_call(msg)
            elif self._cur is None:
                self.errors.append("tool call close outside a tool block")
            else:
                self.errors.append("tool call closed before </function>")
                self._finish_call(msg)

    # -- helpers -------------------------------------------------------------------

    def _emit_outside(self, out: List[str], text: str) -> None:
        if not text:
            return
        if self._calls_seen == 0:
            out.append(text)  # content before the first call
        elif text.strip():
            # Real text after a completed call means the model is rambling
            # past its answer -> stop generation.
            self.status = StreamingStatus.TOOL_CALL_STOP
            self.set_status(self.status)
        # else: whitespace between/after calls -> drop

    def _start_call(self) -> None:
        self._cur_index = self._calls_seen
        self._calls_seen += 1
        self._cur = {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
        self._param_index = 0
        self._state = HEAD
        self._skip_rest = False
        self._frag({
            "index": self._cur_index,
            "id": self._cur["id"],
            "type": "function",
            "function": {"name": "", "arguments": ""},
        })

    def _finish_call(self, msg: Optional[dict]) -> None:
        call = self._cur
        if call is None:
            self._state = OUTSIDE
            self._skip_rest = False
            return
        self._frag({
            "index": self._cur_index,
            "function": {"arguments": "}" if self._param_index > 0 else "{}"},
        })
        self._cur = None
        self._state = OUTSIDE
        self._skip_rest = False
        finished = {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            },
        }
        self._calls.append(finished)
        self._param_index = 0
        if msg is not None:
            # String-keyed object: JsonContainer::concatenate throws on lists.
            msg.setdefault("tool_calls", {})[str(self._cur_index)] = finished
        if self._stop_after:
            self.status = StreamingStatus.TOOL_CALL_STOP
            self.set_status(self.status)
            self._stopped = True

    def _frag(self, payload: Dict[str, Any]) -> None:
        arguments = payload.get("function", {}).get("arguments")
        if arguments is not None and self._cur is not None:
            self._cur["function"]["arguments"] += arguments
        if self.on_fragment is not None:
            self.on_fragment(payload)

    def _consume_value(self, text: str, final: bool) -> None:
        # drop the single wrapper newline right after <parameter=k>
        if not self._value_started and text.startswith("\n"):
            text = text[1:]
        if text:
            self._value_started = True
        # hold back a trailing newline: it may be the wrapper before
        # </parameter>; if more value text follows we flush it then.
        if self._pending_newline:
            text = "\n" + text
            self._pending_newline = False
        if text.endswith("\n"):
            if final:
                text = text[:-1]
            else:
                self._pending_newline = True
                text = text[:-1]
        if text:
            self._param_raw += text
            self._frag({
                "index": self._cur_index,
                "function": {"arguments": _json_escape_fragment(text)},
            })
        if final:
            self._frag({"index": self._cur_index, "function": {"arguments": '"'}})


class Qwen35ToolCallStreamer(StreamerBase):
    """Engine streamer for qwen35 tool requests.

    StreamerBase with its own incremental decode (cumulative decode + delta
    slicing, like ChunkStreamer): boundary tag tokens are matched by ID and
    fed through QwenXMLToolParser with (tag_text, [tag_id]) pairs; everything
    else is decoded incrementally and fed as token-less text deltas. This
    deliberately avoids TextParserStreamer, whose parser chain deadlocks when
    generation runs on a worker thread under asyncio.

    Parsed OpenAI deltas are enqueued on text_queue (same contract as
    ChunkStreamer) wrapped as {"chat_delta": [...]} so the route can
    distinguish them from metrics/error dicts.
    """

    def __init__(self, tokenizer, gen_config):
        super().__init__()
        # enable_thinking=True -> the chat template injects the opening think
        # tag into the prompt, so the stream starts inside a think block.
        # With enable_thinking=False the keep-splitter is disabled entirely.
        thinking = True
        if getattr(gen_config, "chat_template_kwargs", None):
            thinking = bool(gen_config.chat_template_kwargs.get("enable_thinking", True))
        self._reasoning = ReasoningSplitter(enabled=thinking)
        self.tool_parser = QwenXMLToolParser()
        self.tool_parser.on_fragment = self._collect_fragment
        self._fragments: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self._wrapper_ids: Dict[int, str] = {}
        for tag, tag_id in ((TOOL_OPEN, TOOL_OPEN_ID), (TOOL_CLOSE, TOOL_CLOSE_ID)):
            ids = tokenizer.encode(tag).input_ids.data.tolist()[0]
            if len(ids) == 1:
                self._wrapper_ids[ids[0]] = tag
        self.tokens_cache: List[int] = []
        self.last_print_len = 0
        self.text_queue: "asyncio.Queue" = asyncio.Queue()
        self._cancelled = asyncio.Event()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None  # offline use: put_nowait directly

    def _collect_fragment(self, fragment: Dict[str, Any]) -> None:
        self._fragments.append(fragment)

    def _enqueue(self, item) -> None:
        # write()/end() run on the generation thread; asyncio.Queue is not
        # thread-safe, so hop through call_soon_threadsafe when a loop exists.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.text_queue.put_nowait, item)
        else:
            self.text_queue.put_nowait(item)

    def _process_delta(self, text: str, delta_tokens) -> None:
        reason_delta, text_delta = self._reasoning.feed(text)
        content = self.tool_parser.parse({}, text_delta, delta_tokens)
        deltas: List[Dict[str, Any]] = []
        if reason_delta:
            deltas.append({"reasoning_content": reason_delta})
        if content:
            deltas.append({"content": content})
        if self._fragments:
            deltas.append({"tool_calls": self._fragments})
            self._fragments = []
        if deltas:
            self._enqueue({"chat_delta": deltas})

    def _decode_available(self) -> None:
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.last_print_len:
            delta = text[self.last_print_len:]
            if chr(65533) in delta:
                # partial UTF-8 at the boundary; wait for more tokens
                return
            self.last_print_len = len(text)
            self._process_delta(delta, [])

    def _flush_text(self) -> None:
        """Decode and process everything left in the cache, then reset it.

        Called before an atomically-detected boundary token: pre-boundary text
        must be processed before the boundary itself.
        """
        if not self.tokens_cache:
            return
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.last_print_len:
            self._process_delta(text[self.last_print_len:], [])
        self.tokens_cache = []
        self.last_print_len = 0

    def write(self, token) -> StreamingStatus:
        if self._cancelled.is_set():
            self._enqueue(None)
            return StreamingStatus.CANCEL
        ids = token if isinstance(token, list) else [token]
        for tid in ids:
            tag = self._wrapper_ids.get(int(tid))
            if tag is not None:
                self._flush_text()
                self._process_delta(tag, [int(tid)])
            else:
                self.tokens_cache.append(int(tid))
                self._decode_available()
        return self.tool_parser.status

    def end(self) -> None:
        self._flush_text()
        self.tool_parser.finalize()
        if self.tool_parser.errors:
            logger.warning("qwen35 tool parser errors: %s", self.tool_parser.errors)
        self._enqueue(None)

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()


class Qwen35StreamParser:
    """Streaming facade over text deltas (ChunkStreamer path).

    feed() consumes detokenized text (no token IDs available, so boundary IDs
    are synthesized from the tag text) and returns OpenAI delta dicts in order
    [{reasoning_content}, {content}, {tool_calls}], skipping empty entries.
    finish() finalizes the parser (force-closes truncated calls) and returns [].
    """

    def __init__(
        self,
        tools: Optional[List[Dict[str, Any]]] = None,  # accepted; unused
        enable_thinking: bool = True,
    ):
        self._reasoning = ReasoningSplitter(enabled=enable_thinking)
        self._tool_parser = QwenXMLToolParser()
        self._synth = _TextTagSynthesizer()

    def feed(self, text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        reason_delta, text_delta = self._reasoning.feed(text)
        if reason_delta:
            out.append({"reasoning_content": reason_delta})
        content_parts: List[str] = []
        fragments: List[Dict[str, Any]] = []
        for chunk, delta_tokens in self._synth.feed(text_delta):
            c, f = self._parse_chunk(chunk, delta_tokens)
            content_parts.append(c)
            fragments.extend(f)
        content = "".join(content_parts)
        if content:
            out.append({"content": content})
        if fragments:
            out.append({"tool_calls": fragments})
        return out

    def finish(self) -> List[Dict[str, Any]]:
        tail = self._synth.flush()
        if tail:
            self._parse_chunk(tail, None)
        self._tool_parser.finalize()
        if self._tool_parser.errors:
            logger.warning("qwen35 tool parser errors: %s", self._tool_parser.errors)
        return []

    def _parse_chunk(self, chunk: str, delta_tokens):
        fragments: List[Dict[str, Any]] = []

        def sink(fragment: Dict[str, Any]) -> None:
            fragments.append(fragment)

        self._tool_parser.on_fragment = sink
        try:
            content = self._tool_parser.parse({}, chunk, delta_tokens)
        finally:
            self._tool_parser.on_fragment = None
        return content, fragments


def parse_generation(
    text: str,
    tools: Optional[List[Dict[str, Any]]] = None,  # accepted; unused
    enable_thinking: bool = True,
) -> tuple[str, str, Optional[List[Dict[str, Any]]]]:
    """Split model output into (reasoning, content, tool_calls)."""
    reasoning, remainder = ReasoningSplitter(enabled=THINK_CLOSE in text).feed(text)
    if remainder.startswith(THINK_OPEN):
        remainder = remainder[len(THINK_OPEN):]
    if TOOL_OPEN not in remainder:
        return reasoning, remainder, None

    parser = QwenXMLToolParser()
    synth = _TextTagSynthesizer()
    content_parts: List[str] = []
    for chunk, delta_tokens in synth.feed(remainder):
        content_parts.append(parser.parse({}, chunk, delta_tokens))
    tail = synth.flush()
    if tail:
        content_parts.append(parser.parse({}, tail, None))
    parser.finalize()
    if parser.errors:
        logger.debug("qwen35 tool parser errors: %s", parser.errors)
    return reasoning, "".join(content_parts), parser.completed_calls or None


if __name__ == "__main__":
    # Live smoke test for the engine path (Qwen35ToolCallStreamer), mirroring
    # the scratchpad demo: reasoning + tool call in one stream.
    #   python -m src.engine.ov_genai.tool_parse.qwen35 [DEVICE] [MODEL_PATH]
    import sys

    import openvino_genai as ov

    DEVICE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "GPU.0"
    MODEL_PATH = (
        sys.argv[2] if len(sys.argv) > 2 else
        "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen3.5-2B-int4_sym-ov/"
    )
    SMOKE_TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }]

    pipe = ov.VLMPipeline(MODEL_PATH, DEVICE)
    tokenizer = pipe.get_tokenizer()

    from types import SimpleNamespace
    gen_config = SimpleNamespace(
        tools=SMOKE_TOOLS, chat_template_kwargs={"enable_thinking": True}
    )

    streamer = Qwen35ToolCallStreamer(tokenizer, gen_config)

    history = ov.ChatHistory([{"role": "user", "content": "What's the weather in Tokyo and Paris? Use the tool for both cities at once."}])
    history.set_tools(SMOKE_TOOLS)
    history.set_extra_context({"enable_thinking": True})

    config = ov.GenerationConfig()
    config.max_new_tokens = 512

    pipe.generate(history, generation_config=config, streamer=streamer)

    deltas: List[Dict[str, Any]] = []
    while not streamer.text_queue.empty():
        item = streamer.text_queue.get_nowait()
        if item is None:
            break
        if isinstance(item, dict) and "chat_delta" in item:
            deltas.extend(item["chat_delta"])
        else:
            print("queue item:", item)

    print("=" * 60)
    print("PARSED DELTAS")
    print("=" * 60)
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    tool_frags = [f for d in deltas for f in d.get("tool_calls", [])]
    print("reasoning:", repr(reasoning))
    print("content:", repr(content))
    print("fragments:")
    for f in tool_frags:
        print("  ", f)
    args = "".join(
        f["function"]["arguments"]
        for f in tool_frags
        if "arguments" in f.get("function", {})
    )
    names = [f["function"]["name"] for f in tool_frags if f.get("function", {}).get("name")]
    print("names:", names)
    print("arguments:", repr(args))
    try:
        print("arguments JSON:", json.loads(args) if args else None)
    except json.JSONDecodeError as exc:
        print("arguments JSON ERROR:", exc)
    print("parser status:", streamer.tool_parser.get_status())
    print("parser errors:", streamer.tool_parser.errors)
    print("=" * 60)
    per_call: Dict[str, str] = {}
    for f in tool_frags:
        fn = f.get("function", {})
        if "arguments" in fn:
            per_call[str(f["index"])] = per_call.get(str(f["index"]), "") + fn["arguments"]
    ok = True
    for index, text in sorted(per_call.items()):
        try:
            print(f"call {index} args:", json.loads(text))
        except json.JSONDecodeError as exc:
            ok = False
            print(f"call {index} args INVALID: {text!r} ({exc})")
    print("PARALLEL OK" if ok and len(per_call) >= 2 else "NOTE: fewer than 2 calls emitted")

