"""Qwen3.5 XML tool-call parser + streamer built directly on StreamerBase.

Avoids TextParserStreamer/IncrementalParser entirely (a live VLMPipeline run
with a Python TextParserStreamer subclass hung deterministically after tool
call completion). Instead, ToolCallStreamer(StreamerBase) does its own
incremental decode (cumulative decode + delta slicing, like OpenArc's
ChunkStreamer) and feeds text deltas through:

  ReasoningSplitter       - everything before </think> is reasoning_content
                            (Qwen3.5 emits no opening <think> tag)
  QwenXmlToolCallParser   - state machine for the Qwen XML tool format:

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    ...
    </function>
    </tool_call>

Parallel calls are sequential <tool_call> blocks; text may precede the first
block. The parser strips tool XML from content and emits OpenAI-style
streaming fragments:

    {"index": i, "id": ..., "type": "function",
     "function": {"name": "", "arguments": ""}}          # call start
    {"index": i, "function": {"name": NAME}}             # name known
    {"index": i, "function": {"arguments": FRAGMENT}}    # argument deltas

Concatenating all `arguments` fragments for an index yields complete JSON.
Queued delta messages look like:
    {"content": str} / {"reasoning_content": str} / {"tool_calls": [frag, ...]}
None signals completion.
"""
import asyncio
import itertools
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import openvino_genai
from openvino_genai import StreamerBase, StreamingStatus

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
FUNC_OPEN = "<function="
FUNC_CLOSE = "</function>"
PARAM_OPEN = "<parameter="
PARAM_CLOSE = "</parameter>"
GT = ">"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Tags we try to match by token ID (vLLM TokenIDScanner style). Only
# single-token tags qualify; <function= etc. are multi-token ordinary text
# and stay on the text-level hold-back path.
WRAPPER_TAGS = (TOOL_OPEN, TOOL_CLOSE, THINK_OPEN, THINK_CLOSE)

# States
CONTENT = "content"
IN_TOOL_CALL = "in_tool_call"
IN_FUNC_NAME = "in_func_name"
IN_FUNCTION = "in_function"
IN_PARAM_NAME = "in_param_name"
IN_PARAM = "in_param"
AFTER_FUNCTION = "after_function"

REPLACEMENT_CHAR = chr(65533)  # U+FFFD: decode produced a partial character


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
    tag) and terminates it with </think>. Everything before the close tag is
    reasoning; everything after is content.
    """

    def __init__(self, close_tag: str = THINK_CLOSE, enabled: bool = True):
        self._close_tag = close_tag
        self._buf = ""
        # Only treat the stream as reasoning when the prompt actually opened a
        # <think> block (chat template appends '<think>\n' to the prompt when
        # enable_thinking=True; with False it pre-closes an empty block and the
        # stream is pure content).
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


class QwenXmlToolCallParser:
    """Incremental parser for Qwen XML tool calls.

    `tools` is the OpenAI-style tools list from the request; used for argument
    type coercion (booleans arrive Python-style: True/False).
    """

    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None):
        self._param_types: Dict[str, Dict[str, str]] = {}
        for tool in tools or []:
            fn = tool.get("function", {})
            props = (fn.get("parameters") or {}).get("properties") or {}
            self._param_types[fn.get("name", "")] = {
                k: (v.get("type") or "string") for k, v in props.items()
            }
        self.reset()

    def reset(self) -> None:
        self._ids = itertools.count()
        self._buf = ""
        self._state = CONTENT
        self._calls: List[Dict[str, Any]] = []
        self._cur: Optional[Dict[str, Any]] = None   # current call under construction
        self._param_name = ""
        self._param_raw = ""                          # buffered value (non-string types)
        self._param_index = 0                         # params seen in current call
        self._value_started = False                   # first chunk of current value?
        self._pending_newline = False                 # trailing \n hold-back in values
        self.status = StreamingStatus.RUNNING
        self.errors: List[str] = []

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """All calls seen so far, in final OpenAI non-streaming shape."""
        return list(self._calls)

    def feed(self, delta_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Consume a text delta. Returns (content_text, tool_call_fragments)."""
        if not delta_text:
            return "", []
        self._buf += delta_text

        content_out: List[str] = []
        fragments: List[Dict[str, Any]] = []

        while self._buf:
            if not self._step(content_out, fragments):
                break  # need more input

        return "".join(content_out), fragments

    # -- state machine ----------------------------------------------------------

    def _step(self, content_out: list, fragments: list) -> bool:
        """Advance the machine once. False = wait for more input."""
        buf = self._buf

        if self._state == CONTENT:
            idx = buf.find(TOOL_OPEN)
            if idx == -1:
                hold = _holdback_suffix(buf, (TOOL_OPEN,))
                safe, self._buf = buf[: len(buf) - hold], buf[len(buf) - hold:]
                if safe and not self._calls:
                    content_out.append(safe)
                elif safe and safe.strip():
                    # After a completed call, real text means the model is
                    # rambling past its answer -> stop generation.
                    self.status = StreamingStatus.TOOL_CALL_STOP
                # else: trailing whitespace after the final call -> drop
                return False  # remaining buf is a partial tag prefix; wait
            if idx > 0:
                pre, self._buf = buf[:idx], buf[idx:]
                if not self._calls:
                    content_out.append(pre)
                elif pre.strip():
                    self.status = StreamingStatus.TOOL_CALL_STOP
                return True
            self._buf = buf[len(TOOL_OPEN):]
            self._start_call()
            self._state = IN_TOOL_CALL
            return True

        if self._state == IN_TOOL_CALL:
            return self._expect_tag(buf, (FUNC_OPEN,), IN_FUNC_NAME)

        if self._state == IN_FUNC_NAME:
            gt = buf.find(GT)
            if gt == -1:
                if "<" in buf:  # names never contain '<'; bail on garbage
                    self.errors.append(f"malformed function name: {buf!r}")
                    self._state = CONTENT
                    return True
                return False
            name = buf[:gt].strip()
            self._buf = buf[gt + 1:]
            self._register_call(name, fragments)
            self._state = IN_FUNCTION
            return True

        if self._state == IN_FUNCTION:
            stripped = buf.lstrip()
            if stripped.startswith(PARAM_OPEN):
                self._buf = stripped[len(PARAM_OPEN):]
                self._state = IN_PARAM_NAME
                return True
            if stripped.startswith(FUNC_CLOSE):
                self._buf = stripped[len(FUNC_CLOSE):]
                self._close_call(fragments)
                self._state = AFTER_FUNCTION
                return True
            if not stripped or any(
                t.startswith(stripped) and t != stripped
                for t in (PARAM_OPEN, FUNC_CLOSE)
            ):
                return False  # whitespace only, or partial tag
            self.errors.append(f"unexpected text inside <function>: {stripped[:20]!r}")
            self._buf = stripped[1:]
            return True

        if self._state == IN_PARAM_NAME:
            gt = buf.find(GT)
            if gt == -1:
                if "<" in buf:
                    self.errors.append(f"malformed parameter name: {buf!r}")
                    self._state = IN_FUNCTION
                    return True
                return False
            self._param_name = buf[:gt].strip()
            self._buf = buf[gt + 1:]
            self._param_raw = ""
            self._value_started = False
            self._pending_newline = False
            self._state = IN_PARAM
            # open the JSON member
            ptype = self._param_type(self._cur["function"]["name"], self._param_name)
            sep = "{" if self._param_index == 0 else ", "
            self._param_index += 1
            self._frag(fragments, f'{sep}{json.dumps(self._param_name)}: ')
            if ptype == "string":
                self._frag(fragments, '"')
            return True

        if self._state == IN_PARAM:
            idx = buf.find(PARAM_CLOSE)
            ptype = self._param_type(self._cur["function"]["name"], self._param_name)
            if idx == -1:
                hold = _holdback_suffix(buf, (PARAM_CLOSE,))
                safe, self._buf = buf[: len(buf) - hold], buf[len(buf) - hold:]
                if safe:
                    self._consume_value(safe, fragments, ptype, final=False)
                return False
            value, self._buf = buf[:idx], buf[idx + len(PARAM_CLOSE):]
            self._consume_value(value, fragments, ptype, final=True)
            self._state = IN_FUNCTION
            return True

        if self._state == AFTER_FUNCTION:
            return self._expect_tag(buf, (TOOL_CLOSE,), CONTENT)

        return False

    def _expect_tag(self, buf: str, tags, next_state: str) -> bool:
        stripped = buf.lstrip()
        for t in tags:
            if stripped.startswith(t):
                self._buf = stripped[len(t):]
                self._state = next_state
                return True
        if not stripped or any(
            t.startswith(stripped) and t != stripped for t in tags
        ):
            return False  # whitespace only, or partial tag
        self.errors.append(f"expected {tags}, got {stripped[:20]!r}")
        self._buf = stripped[1:]
        return True

    # -- call/fragment helpers ----------------------------------------------------

    def _frag(self, fragments: list, arguments: str):
        fragments.append({
            "index": len(self._calls) - 1,
            "function": {"arguments": arguments},
        })
        self._cur["function"]["arguments"] += arguments

    def _start_call(self):
        # Pending only: not registered in `_calls` (and no fragment emitted)
        # until `_register_call` confirms this is really Qwen's
        # <function=NAME> syntax. A <tool_call> block using a different
        # format (e.g. Hermes-style raw JSON) never registers, so it can't
        # surface as a bogus half-empty tool call - it's left for the
        # accumulated-text Hermes fallback in openai.py to pick up instead.
        self._cur = {
            "id": f"call_{next(self._ids):024x}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
        self._param_index = 0

    def _register_call(self, name: str, fragments: list):
        self._cur["function"]["name"] = name
        self._calls.append(self._cur)
        fragments.append({
            "index": len(self._calls) - 1,
            "id": self._cur["id"],
            "type": "function",
            "function": {"name": name, "arguments": ""},
        })

    def _close_call(self, fragments: list):
        self._frag(fragments, "}" if self._param_index > 0 else "{}")
        self._cur["done"] = True

    def _param_type(self, func_name: str, param: str) -> str:
        return self._param_types.get(func_name, {}).get(param, "string")

    def _consume_value(self, text: str, fragments: list, ptype: str, final: bool):
        # drop the single wrapper newline right after <parameter=k>
        if not self._value_started and text.startswith("\n"):
            text = text[1:]
        if text:
            self._value_started = True
        if ptype == "string":
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
                self._frag(fragments, _json_escape_fragment(text))
            if final:
                self._frag(fragments, '"')
        else:
            # buffer short scalar values; coerce + emit once complete
            self._param_raw += text
            if final:
                self._frag(fragments, self._coerce(self._param_raw, ptype))

    def _coerce(self, raw: str, ptype: str) -> str:
        """Serialize a raw XML parameter value as JSON text."""
        v = raw.strip()
        if ptype in ("integer", "number"):
            return v  # model emits digits
        if ptype == "boolean":
            low = v.lower()
            if low in ("true", "1", "yes"):
                return "true"
            if low in ("false", "0", "no"):
                return "false"
            return json.dumps(v)  # fallback: keep as string
        if ptype in ("array", "object"):
            try:
                return json.dumps(json.loads(v))
            except (json.JSONDecodeError, TypeError):
                return json.dumps(v)
        return json.dumps(raw.strip("\n"))

    def finalize(self) -> None:
        """Call at end of generation to surface truncated structures."""
        if self._cur is not None and not self._cur.get("done"):
            self.errors.append("generation ended with unterminated tool call")
        if self._state != CONTENT:
            self.errors.append(f"generation ended in state {self._state}")


class ToolCallStreamer(StreamerBase):
    """StreamerBase doing its own incremental decode + tool-call parsing.

    Mirrors ChunkStreamer's contract: decoded/parsed delta messages are put on
    `queue` from the generation thread; None signals completion. Return value
    controls generation: TOOL_CALL_STOP when the model rambles past a closed
    tool call, CANCEL after cancel().
    """

    def __init__(self, tokenizer, tools: Optional[list] = None,
                 enable_thinking: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.reasoning = ReasoningSplitter(enabled=enable_thinking)
        self.tool_parser = QwenXmlToolCallParser(tools)
        self.tokens_cache: List[int] = []
        self.last_print_len = 0
        self._wrapper_ids: Dict[int, str] = {}
        for tag in WRAPPER_TAGS:
            ids = tokenizer.encode(tag).input_ids.data.tolist()[0]
            if len(ids) == 1:
                self._wrapper_ids[ids[0]] = tag
        self.queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
        self._cancelled = asyncio.Event()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None  # offline use: no loop, put_nowait directly

    def _enqueue(self, msg) -> None:
        # write()/end() run on the C++ generation thread; asyncio.Queue is not
        # thread-safe, so hop through call_soon_threadsafe when a loop exists.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, msg)
        else:
            self.queue.put_nowait(msg)

    def _process_delta(self, delta: str) -> None:
        reasoning, text = self.reasoning.feed(delta)
        content, fragments = self.tool_parser.feed(text)
        msg: Dict[str, Any] = {}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if content:
            msg["content"] = content
        if fragments:
            msg["tool_calls"] = fragments
        if msg:
            self._enqueue(msg)

    def _decode_available(self) -> None:
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.last_print_len:
            delta = text[self.last_print_len:]
            if REPLACEMENT_CHAR in delta:
                # partial UTF-8 at the boundary; wait for more tokens
                return
            self.last_print_len = len(text)
            self._process_delta(delta)

    def _flush_text(self) -> None:
        """Decode and process everything left in the cache, then reset it.

        Called before an atomically-detected wrapper token: the special token
        is a hard boundary, so any held-back partial tag in the text parser
        must resolve now (it can't be a prefix of a wrapper tag anymore).
        """
        if not self.tokens_cache:
            return
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.last_print_len:
            self._process_delta(text[self.last_print_len:])
        self.tokens_cache = []
        self.last_print_len = 0

    def write(self, token: Union[int, List[int]]) -> StreamingStatus:
        if self._cancelled.is_set():
            self._enqueue(None)
            return StreamingStatus.CANCEL

        ids = token if isinstance(token, list) else [token]
        for tid in ids:
            tag = self._wrapper_ids.get(int(tid))
            if tag is not None:
                self._flush_text()
                if tag == THINK_OPEN:
                    # Opening tag is baked into the prompt by the chat
                    # template; a stray one in the stream is a no-op.
                    continue
                # Atomic delivery: hold-back in the text parser resolves
                # immediately, so this can never straddle chunks.
                self._process_delta(tag)
            else:
                self.tokens_cache.append(int(tid))
                self._decode_available()

        return self.tool_parser.status

    def end(self) -> None:
        self._flush_text()
        self.tool_parser.finalize()
        self._enqueue(None)

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()
