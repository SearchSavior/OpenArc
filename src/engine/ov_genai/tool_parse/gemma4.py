"""Gemma 4 tool-call parser (Chimera-X-26B-A4B and other Gemma-4-derived exports).

Unlike Qwen3.5, every Gemma 4 protocol tag is special=True, so the tags NEVER
appear in decoded text; they exist only as token IDs. Token-ID boundary
detection is mandatory, not just preferred:

  - Gemma4ToolCallStreamer sees raw ids in write() and intercepts the protocol
    ids (they would be stripped from its own incremental decode otherwise);
    the payload text ('call:name{key:value, ...}') arrives token-less between
    boundaries.
  - Quote tokens (<|"|>, id 52) are special too, so argument strings arrive
    UNQUOTED (city:Paris, not city:"Paris"). Values are emitted as JSON
    strings, no schema-driven type coercion (same decision as qwen35).

Assistant protocol (chat_template.jinja):
  reasoning : <|channel>thought\n <text> \n<channel|>
  tool call : <|tool_call>call:<name>{<key>:<value>, ...}<tool_call|>
  turn end  : <turn|> (eos_token_id 106)

Consumers:
  Gemma4ToolCallStreamer - StreamerBase subclass used by the engine for
                           streaming requests; does its own incremental decode
                           and enqueues {"chat_delta": [...]} dicts. Deliberately
                           avoids TextParserStreamer, which deadlocks when
                           generate runs on a worker thread under asyncio.
  parse_generation()     - raw-text splitter for non-streaming requests (the
                           engine decodes with skip_special_tokens=False for
                           gemma4-parser models so the tags survive as text).

Parsed OpenAI streaming fragments (same contract as qwen35):
    {"index": i, "id": ..., "type": "function",
     "function": {"name": "", "arguments": ""}}          # call start
    {"index": i, "function": {"name": NAME}}             # name known
    {"index": i, "function": {"arguments": FRAGMENT}}    # argument deltas

Concatenating all `arguments` fragments for an index yields complete JSON.
"""
import asyncio
import json
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from openvino_genai import IncrementalParser, StreamerBase, StreamingStatus

from src.engine.ov_genai.tool_parse.qwen35 import _is_partial, _json_escape_fragment

logger = logging.getLogger(__name__)

# Tag text, used only by the raw-text path (parse_generation). Invisible to
# the token-ID path: all of these are special=True tokens.
TOOL_OPEN = "<|tool_call>"
TOOL_CLOSE = "<tool_call|>"
CHANNEL_OPEN = "<|channel>"
CHANNEL_CLOSE = "<channel|>"
TURN_END = "<turn|>"
EOS = "<eos>"
CALL_PREFIX = "call:"

# Protocol token IDs (validated against the Chimera-X-26B tokenizer; the
# streamer re-checks them via encode lookups at construction time).
TOOL_OPEN_ID = 48
TOOL_CLOSE_ID = 49
CHANNEL_OPEN_ID = 100
CHANNEL_CLOSE_ID = 101
TOOL_BOUNDARY_IDS = (TOOL_OPEN_ID, TOOL_CLOSE_ID)
THOUGHT_HEADER = "thought"

# Parser states
OUTSIDE = "outside"   # no open block: text is content (or ramble after a call)
HEAD = "head"         # awaiting 'call:'
NAME = "name"         # awaiting '{' that ends the function name
ARGS_KEY = "args_key"  # awaiting 'key:' (or '}' for a zero-argument call)
VALUE = "value"       # inside an argument value (brace-depth tracked)
TAIL = "tail"         # payload closed, awaiting the close boundary

_PAYLOAD_RE = re.compile(r"call:([A-Za-z_][\w]*)\{(.*)\}", re.DOTALL)


def wants_engine_stream(tools: Optional[List[Dict[str, Any]]], thinking_enabled: bool) -> bool:
    """True when a gemma4 request must use the token-ID engine streamer.

    Thought-channel tags are special=True, so a plain ChunkStreamer text path
    cannot split reasoning; the engine streamer is required whenever tools are
    requested OR thinking is enabled. With both off the model answers with
    plain content (the template injects an empty thought channel) and
    ChunkStreamer suffices.
    """
    return bool(tools) or bool(thinking_enabled)


class Gemma4ChannelSplitter:
    """Splits Gemma 4 thought channels from content by token ID.

    All channel tags are special=True, so boundaries are ID-only: a
    CHANNEL_OPEN_ID opens a channel, CHANNEL_CLOSE_ID closes it. The visible
    'thought\\n' header text is consumed so it does not leak into content;
    channels with other names pass through as content without their wrapper
    tags (conservative). Thought text streams out as reasoning deltas with a
    one-line tail held back so the newline before the close tag is trimmed.

    The driver contract is exact attribution: each feed() call either carries
    text (no protocol ids) or protocol ids (empty text).
    """

    def __init__(self):
        self._state = "content"    # content | header | thought | opaque
        self._header = ""          # accumulated channel-name line
        self._reasoning = ""       # held reasoning tail

    def feed(self, text: str, token_ids) -> Tuple[str, str, List[int]]:
        """Returns (reasoning_delta, content_delta, passthrough_token_ids)."""
        reason_parts: List[str] = []
        content_parts: List[str] = []
        passthrough: List[int] = []
        for token in (token_ids or []):
            tid = int(token)
            if tid == CHANNEL_OPEN_ID and self._state == "content":
                self._state = "header"
                self._header = ""
                continue
            if tid == CHANNEL_CLOSE_ID and self._state in ("thought", "opaque"):
                if self._state == "thought":
                    tail = self._reasoning.rstrip("\n")
                    if tail:
                        reason_parts.append(tail)
                    self._reasoning = ""
                self._state = "content"
                continue
            passthrough.append(tid)
        if text:
            if self._state == "header":
                self._header += text
                if "\n" in self._header:
                    line, _, rest = self._header.partition("\n")
                    self._header = ""
                    if line.strip() == THOUGHT_HEADER:
                        self._state = "thought"
                        self._reasoning += rest
                    else:
                        self._state = "opaque"
                        content_parts.append(line + "\n" + rest)
            elif self._state == "thought":
                # Stream thought text out, keeping a one-line tail so the
                # newline before the close tag can be trimmed cleanly.
                self._reasoning += text
                cut = self._reasoning.rfind("\n")
                if cut > 0:
                    reason_parts.append(self._reasoning[:cut])
                    self._reasoning = self._reasoning[cut:]
            else:  # content | opaque
                content_parts.append(text)
        return "".join(reason_parts), "".join(content_parts), passthrough


class Gemma4ToolCallParser(IncrementalParser):
    """IncrementalParser for Gemma 4 tool calls, bound by token IDs.

    Payload grammar: 'call:<name>{<key>:<value>, ...}' where values may be
    nested brace objects. Boundary detection is token-ID only (the tag text
    never appears in the stream); payload text is streamed through the state
    machine and emitted as incremental OpenAI argument fragments:

        'call:get_weather{city:Paris, unit:{...}}' ->
            {"index": 0, "id": ..., "type": "function",
             "function": {"name": "", "arguments": ""}}
            {"index": 0, "function": {"name": "get_weather"}}
            {"index": 0, "function": {"arguments": '{"city": "'}}
            {"index": 0, "function": {"arguments": 'Paris"}}
            {"index": 0, "function": {"arguments": ', "unit": "'}}
            ...

    Values are JSON strings (unquoted in the payload because quote tokens are
    special=True). With stop_after_tool_call=True the parser requests
    StreamingStatus.TOOL_CALL_STOP right after each complete call; by default
    (False) sequential parallel calls are allowed and generation is only
    stopped when real text follows a completed call.
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
        self._calls_seen = 0                         # calls started (open boundary seen)
        self._calls: List[Dict[str, Any]] = []       # completed calls, OpenAI shape
        self._param_index = 0
        self._depth = 0
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
            if self._state == VALUE:
                self._end_value()
            self._finish_call({})
        elif self._state != OUTSIDE:
            self.errors.append(f"generation ended in state {self._state}")

    # -- IncrementalParser API ---------------------------------------------------

    def parse(self, msg: dict, delta_text: str, delta_tokens=None) -> str:
        """Consume one decoded delta. Returns content text for this delta.

        Mutates `msg` with completed calls under msg["tool_calls"][str(index)]
        and emits streaming fragments through `on_fragment`. The driver
        contract is that deltas either carry text (no protocol ids) or
        protocol ids (empty text); if both arrive, text is processed first.
        """
        if self._stopped:
            return ""
        out: List[str] = []
        if delta_text:
            self._pending += delta_text
            self._machine(out, final=bool(delta_tokens))
        for token in (delta_tokens or []):
            tid = int(token)
            if tid in TOOL_BOUNDARY_IDS:
                self._machine(out, final=True)
                self._apply_boundary(tid, msg)
                if self._stopped:
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
                        f"unexpected text after payload: {self._pending[:20]!r}"
                    )
                self._pending = ""
                return
            if self._state == HEAD:
                stripped = self._pending.lstrip()
                self._pending = stripped
                if stripped.startswith(CALL_PREFIX):
                    self._pending = stripped[len(CALL_PREFIX):]
                    self._state = NAME
                    continue
                if not stripped:
                    return
                if not final and _is_partial(stripped, (CALL_PREFIX,)):
                    return
                self.errors.append(
                    f"expected 'call:' after tool open, got {stripped[:20]!r}"
                )
                self._skip_rest = True
                self._pending = ""
                return
            if self._state == NAME:
                brace = self._pending.find("{")
                if brace == -1:
                    if final or "}" in self._pending:
                        self.errors.append(
                            f"malformed tool call name: {self._pending[:20]!r}"
                        )
                        self._skip_rest = True
                        self._pending = ""
                    return
                name = self._pending[:brace].strip()
                self._pending = self._pending[brace + 1:]
                if not name:
                    self.errors.append("empty tool call name")
                    self._skip_rest = True
                    self._pending = ""
                    return
                if self._cur is not None:
                    self._cur["function"]["name"] = name
                    self._frag({"index": self._cur_index, "function": {"name": name}})
                self._state = ARGS_KEY
                continue
            if self._state == ARGS_KEY:
                i = 0
                n = len(self._pending)
                while i < n:
                    ch = self._pending[i]
                    if ch == ":":
                        key = self._pending[:i].strip()
                        self._pending = self._pending[i + 1:]
                        if not key:
                            self.errors.append("empty argument key")
                            self._skip_rest = True
                            self._pending = ""
                            return
                        self._begin_param(key)
                        self._state = VALUE
                        self._depth = 0
                        break
                    if ch == "}":
                        pre = self._pending[:i].strip()
                        self._pending = self._pending[i + 1:]
                        if pre:
                            self.errors.append(
                                f"trailing text before payload end: {pre[:20]!r}"
                            )
                        self._state = TAIL
                        break
                    i += 1
                else:
                    if final and self._pending.strip():
                        self.errors.append(
                            f"malformed argument list: {self._pending[:20]!r}"
                        )
                        self._skip_rest = True
                        self._pending = ""
                    return
                continue
            if self._state == VALUE:
                i = 0
                n = len(self._pending)
                while i < n:
                    ch = self._pending[i]
                    if ch == "{":
                        self._depth += 1
                    elif ch == "}":
                        if self._depth == 0:
                            self._consume_value_chars(self._pending[:i])
                            self._pending = self._pending[i + 1:]
                            self._end_value()
                            self._state = TAIL
                            break
                        self._depth -= 1
                    elif ch == "," and self._depth == 0:
                        self._consume_value_chars(self._pending[:i])
                        self._pending = self._pending[i + 1:]
                        self._end_value()
                        self._state = ARGS_KEY
                        break
                    i += 1
                else:
                    # No boundary char in this segment; everything is value text.
                    self._consume_value_chars(self._pending)
                    self._pending = ""
                    return
                continue

    # -- boundaries ----------------------------------------------------------------

    def _apply_boundary(self, tag_id: int, msg: Optional[dict] = None) -> None:
        if tag_id == TOOL_OPEN_ID:
            if self._cur is not None:
                self.errors.append("new tool call opened before previous closed")
                self._close_open_value()
                self._finish_call(msg if msg is not None else {})
            self._start_call()
        else:
            if self._cur is None:
                self.errors.append("tool call close outside a tool block")
            elif self._state == TAIL:
                self._finish_call(msg)
            else:
                self.errors.append(
                    f"tool call closed mid-payload (state {self._state})"
                )
                self._close_open_value()
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
        self._depth = 0
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

    def _begin_param(self, key: str) -> None:
        self._param_index += 1
        sep = "{" if self._param_index == 1 else ", "
        self._frag({
            "index": self._cur_index,
            "function": {"arguments": f"{sep}{json.dumps(key)}: "},
        })
        self._frag({"index": self._cur_index, "function": {"arguments": '"'}})

    def _consume_value_chars(self, text: str) -> None:
        if text:
            self._frag({
                "index": self._cur_index,
                "function": {"arguments": _json_escape_fragment(text)},
            })

    def _end_value(self) -> None:
        self._frag({"index": self._cur_index, "function": {"arguments": '"'}})

    def _close_open_value(self) -> None:
        if self._state == VALUE:
            self._end_value()


class Gemma4ToolCallStreamer(StreamerBase):
    """Engine streamer for gemma4 tool requests.

    StreamerBase with its own incremental decode (cumulative decode + delta
    slicing, like ChunkStreamer): protocol tag tokens are matched by ID and
    fed through the channel splitter / tool parser as ID-only events, while
    everything else is decoded incrementally and fed as token-less text
    deltas. Special=True tags vanish from decode(), which is exactly why they
    must be intercepted here. This deliberately avoids TextParserStreamer,
    whose parser chain deadlocks when generation runs on a worker thread
    under asyncio.

    Parsed OpenAI deltas are enqueued on text_queue (same contract as
    ChunkStreamer) wrapped as {"chat_delta": [...]} so the route can
    distinguish them from metrics/error dicts.
    """

    def __init__(self, tokenizer, gen_config):
        super().__init__()
        self._channels = Gemma4ChannelSplitter()
        self.tool_parser = Gemma4ToolCallParser()
        self.tool_parser.on_fragment = self._collect_fragment
        self._fragments: List[Dict[str, Any]] = []
        # Raw tagged-output reconstruction: decoded text deltas plus the tag
        # text of every intercepted protocol id. Lets non-streaming callers
        # obtain the raw text parse_generation expects without relying on
        # pipeline-level skip_special_tokens (not available on this wheel).
        self._raw_parts: List[str] = []
        self.tokenizer = tokenizer
        self._protocol_ids: Dict[int, int] = {}
        self._protocol_tags: Dict[int, str] = {}
        for tag, tag_id in (
            (TOOL_OPEN, TOOL_OPEN_ID),
            (TOOL_CLOSE, TOOL_CLOSE_ID),
            (CHANNEL_OPEN, CHANNEL_OPEN_ID),
            (CHANNEL_CLOSE, CHANNEL_CLOSE_ID),
        ):
            ids = tokenizer.encode(tag).input_ids.data.tolist()[0]
            if len(ids) == 1:
                if ids[0] != tag_id:
                    logger.warning(
                        "gemma4 tag %r encodes to id %d, expected %d",
                        tag, ids[0], tag_id,
                    )
                self._protocol_ids[ids[0]] = tag_id
                self._protocol_tags[tag_id] = tag
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
        self._raw_parts.append(text)
        reason_delta, text_delta, content_ids = self._channels.feed(text, delta_tokens)
        content = self.tool_parser.parse({}, text_delta, content_ids)
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

        Called before an intercepted protocol token: pre-boundary text must be
        processed before the boundary itself.
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
            protocol_id = self._protocol_ids.get(int(tid))
            if protocol_id is not None:
                self._flush_text()
                self._raw_parts.append(self._protocol_tags[protocol_id])
                self._process_delta("", [protocol_id])
            else:
                self.tokens_cache.append(int(tid))
                self._decode_available()
        return self.tool_parser.status

    @property
    def raw_text(self) -> str:
        """Reconstructed raw tagged output (protocol tags as literal text)."""
        return "".join(self._raw_parts)

    def end(self) -> None:
        self._flush_text()
        self.tool_parser.finalize()
        if self.tool_parser.errors:
            logger.warning("gemma4 tool parser errors: %s", self.tool_parser.errors)
        self._enqueue(None)

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()


# -- raw-text path (non-streaming) ------------------------------------------------


def _split_args(args_text: str) -> Dict[str, str]:
    """Split 'key:value,key:{nested:{...}},...' at top level (brace-depth 0).

    Values stay raw strings (unquoted in the payload; quote tokens are
    special=True, so no type information survives anyway).
    """
    parts: List[str] = []
    depth = 0
    current = ""
    for ch in args_text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current:
        parts.append(current)
    arguments: Dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        arguments[key.strip()] = value.strip()
    return arguments


def _parse_payload(payload: str) -> Optional[Dict[str, Any]]:
    """Parse a complete 'call:<name>{...}' payload into an OpenAI tool call."""
    match = _PAYLOAD_RE.fullmatch(payload.strip())
    if not match:
        return None
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": match.group(1),
            "arguments": json.dumps(_split_args(match.group(2))),
        },
    }


def _split_channels_text(text: str) -> Tuple[str, str]:
    """Extract thought channels from raw tagged text.

    Returns (reasoning, remainder). Thought-channel bodies are rstripped of
    the newline before the close tag; channels with other names pass through
    as content without their wrapper tags; unterminated channels are kept
    verbatim in the remainder.
    """
    reasoning_parts: List[str] = []
    out: List[str] = []
    pos = 0
    while True:
        i = text.find(CHANNEL_OPEN, pos)
        if i == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:i])
        j = text.find(CHANNEL_CLOSE, i + len(CHANNEL_OPEN))
        if j == -1:
            out.append(text[i:])
            break
        block = text[i + len(CHANNEL_OPEN):j]
        header, sep, body = block.partition("\n")
        if sep and header.strip() == THOUGHT_HEADER:
            reasoning_parts.append(body.rstrip("\n"))
        else:
            out.append(block)
        pos = j + len(CHANNEL_CLOSE)
    return "".join(reasoning_parts), "".join(out)


def parse_generation(
    text: str,
    tools: Optional[List[Dict[str, Any]]] = None,  # accepted; unused
    enable_thinking: bool = True,  # accepted for dispatch symmetry; unused
) -> tuple[str, str, Optional[List[Dict[str, Any]]]]:
    """Split raw tagged model output into (reasoning, content, tool_calls).

    Expects text decoded with skip_special_tokens=False (the engine does this
    for gemma4-parser models) so <|channel>/<|tool_call> tags survive as
    literal text.
    """
    if CHANNEL_OPEN not in text and TOOL_OPEN not in text:
        return "", text, None
    cleaned = text.replace(TURN_END, "").replace(EOS, "")
    reasoning, rest = _split_channels_text(cleaned)
    calls: List[Dict[str, Any]] = []
    content_parts: List[str] = []
    pos = 0
    while True:
        i = rest.find(TOOL_OPEN, pos)
        if i == -1:
            content_parts.append(rest[pos:])
            break
        content_parts.append(rest[pos:i])
        j = rest.find(TOOL_CLOSE, i + len(TOOL_OPEN))
        if j == -1:
            # Unterminated block: keep the raw text as content.
            content_parts.append(rest[i:])
            break
        payload = rest[i + len(TOOL_OPEN):j]
        call = _parse_payload(payload)
        if call is not None:
            calls.append(call)
        else:
            logger.debug("gemma4 malformed tool call payload: %r", payload[:80])
        pos = j + len(TOOL_CLOSE)
    return reasoning, "".join(content_parts).strip(), calls or None


if __name__ == "__main__":
    # Live smoke test for the engine path (Gemma4ToolCallStreamer) on the
    # Chimera-X-26B VLM: reasoning channel + parallel tool calls in one stream.
    #   python -m src.engine.ov_genai.tool_parse.gemma4 [DEVICE] [MODEL_PATH]
    import sys

    import openvino_genai as ov

    DEVICE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "GPU.0"
    MODEL_PATH = (
        sys.argv[2] if len(sys.argv) > 2 else
        "/mnt/Ironwolf-4TB/Models/OpenVINO/Gemma/Chimera-X-26B-A4B-int4-ov/"
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

    streamer = Gemma4ToolCallStreamer(tokenizer, gen_config)

    history = ov.ChatHistory([
        {"role": "user", "content": "What's the weather in Tokyo and Paris? Use the tool for both cities at once."}
    ])
    history.set_tools(SMOKE_TOOLS)
    history.set_extra_context({"enable_thinking": True})

    config = ov.GenerationConfig()
    config.max_new_tokens = 1024

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
