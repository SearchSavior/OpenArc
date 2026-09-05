"""Muse-Glimmer (Harmony-style) tool-call parser for MuseGlimmerForConditionalGeneration.

Assistant output uses `to=` channel routing over Harmony control tokens, with
Claude-style 'atem' XML tool-call payloads (chat_template.jinja):

    reasoning : [<|start|>assistant] to=self<|message|>{text}<|eom|>
    tool call : [<|start|>assistant] to=NAME<|message|><atem:function_calls>
                          <atem:invoke name="NAME">
                          <atem:parameter name="KEY">VALUE</atem:parameter>
                          ...
                          </atem:invoke>
                          </atem:function_calls><|eom|>|<|eot|>
    content   : [<|start|>assistant] to=user<|message|>{text}<|eot|>

The generation prompt ends with '<|start|>assistant', so the first message's
header (' to=...') arrives without the start token; subsequent messages within
one generation re-emit '<|start|>assistant to=...'.

Every control token is special=True, so it NEVER appears in decoded text and
boundaries are token-ID only:

    <|start|>=200022  <|message|>=200023  <|eom|>=200007  <|eot|>=200008
    <|end_of_text|>=200001

'atem' tags are ordinary multi-token text and are matched at the text level
with prefix hold-back (like qwen35's <function=/</function> tags).

Consumers:
  MuseGlimmerToolCallStreamer - StreamerBase subclass used by the engine for
                                streaming requests; does its own incremental
                                decode, intercepts control tokens by ID, and
                                enqueues {"chat_delta": [...]} dicts. Deliberately
                                avoids TextParserStreamer, which deadlocks when
                                generate runs on a worker thread under asyncio.
  parse_generation()          - raw-text splitter for non-streaming requests
                                (expects text with literal control tokens, e.g.
                                MuseGlimmerToolCallStreamer.raw_text; the VLM
                                decode always strips specials on this wheel).

wants_engine_stream() is always True: even a plain content response opens with
' to=user<|message|>' header text, which a text-only path cannot strip (the
<|message|> token is special=True and invisible to decoded text).

Parsed OpenAI streaming fragments (same contract as qwen35/gemma4):
    {"index": i, "id": ..., "type": "function",
     "function": {"name": "", "arguments": ""}}          # call start
    {"index": i, "function": {"name": NAME}}             # name known / corrected
    {"index": i, "function": {"arguments": FRAGMENT}}    # argument deltas

Concatenating all `arguments` fragments for an index yields complete JSON.
Parameter values are emitted as JSON strings (raw payload text, no
schema-driven type coercion) - the same decision as qwen35 and gemma4.

Recipients/names are normalized against the requested tool names: 'tools.X'
(streaming recipient 'tools.*') maps back to the registered name 'X' when X
matches; unknown names surface verbatim.
"""
import asyncio
import json
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from openvino_genai import IncrementalParser, StreamerBase, StreamingStatus

from src.engine.ov_genai.tool_parse.qwen35 import (
    _holdback_suffix,
    _json_escape_fragment,
)

logger = logging.getLogger(__name__)

# -- Harmony control tokens (special=True; boundaries by token ID only) --------
START = "<|start|>"
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"
EOS = "<|end_of_text|>"

START_ID = 200022
MESSAGE_ID = 200023
EOM_ID = 200007
EOT_ID = 200008
EOS_ID = 200001
CHANNEL_END_IDS = (EOM_ID, EOT_ID, EOS_ID)

# -- atem payload tags (ordinary multi-token text; matched with hold-back) -----
FC_OPEN = "<atem:function_calls>"
FC_CLOSE = "</atem:function_calls>"
INVOKE_OPEN = '<atem:invoke name="'
INVOKE_CLOSE = "</atem:invoke>"
PARAM_OPEN = '<atem:parameter name="'
PARAM_CLOSE = "</atem:parameter>"
ATTR_END = '">'  # terminates both the invoke name and the parameter name attrs

# Every atem tag: hold-back set so partial tags are never emitted as content.
HOLD_TAGS = (FC_OPEN, FC_CLOSE, INVOKE_OPEN, INVOKE_CLOSE, PARAM_OPEN, PARAM_CLOSE)

BOS_ID = 200000  # <|begin_of_text|>; Tokenizer.encode() prepends it to every encode

RECIPIENT_RE = re.compile(r"\bto=([^\s]+)")
SELF = "self"
USER = "user"

# Safety valve: a header that never receives <|message|> degrades to content
# once it grows past this (keeps a malformed response streaming).
HEADER_LIMIT = 256

# Parser states
OUTSIDE = "outside"      # in a tool channel, between tags: pre-block text or ramble
BLOCK = "block"          # inside <atem:function_calls>, awaiting <atem:invoke ...
INVOKE_NAME = "invoke_name"  # awaiting '">' that ends the invoke name attr
BODY = "body"            # inside <atem:invoke>: params or </atem:invoke>
PARAM_NAME = "param_name"  # awaiting '">' that ends the parameter name attr
VALUE = "value"          # inside a parameter value, verbatim until </atem:parameter>


def wants_engine_stream(
    tools: Optional[List[Dict[str, Any]]],
    thinking_enabled: bool,  # accepted for dispatch symmetry; unused
) -> bool:
    """True: every museglimmer request must use the token-ID engine streamer.

    Channel routing is Harmony token driven (<|start|>/<|message|> are
    special=True and stripped from decoded text), so even a plain content
    response opens with ' to=user' header text that only the engine streamer
    can strip. Reasoning ('to=self') and tool channels are ID-only for the
    same reason.
    """
    return True


def _find_any(text: str, tags) -> Tuple[int, Optional[str]]:
    """Earliest (index, tag) occurrence of any full tag in text, or (-1, None)."""
    best_idx, best_tag = -1, None
    for tag in tags:
        idx = text.find(tag)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_tag = idx, tag
    return best_idx, best_tag


class MuseGlimmerToolParser(IncrementalParser):
    """IncrementalParser for the 'atem' tool-call payload, text-bound.

    Call lifecycle: a tool channel opens via begin_call(RECIPIENT) (called by
    the channel splitter when '<|message|>' resolves a non-self/user recipient),
    the payload streams through the state machine, and the call finishes at
    '</atem:invoke>' or at the channel end ('<|eom|>'/'<|eot|>' forwarded as
    delta_tokens by the splitter). Values are raw payload text emitted as
    incremental JSON-string argument fragments. With stop_after_tool_call=True
    the parser requests StreamingStatus.TOOL_CALL_STOP right after the first
    complete call; by default (False) sequential parallel calls are allowed and
    generation is only stopped when real text follows a completed call.
    """

    def __init__(
        self,
        on_fragment: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_after_tool_call: bool = False,
        tool_names: Optional[set] = None,
    ):
        super().__init__()
        self.on_fragment = on_fragment
        self._stop_after = stop_after_tool_call
        self._tool_names = set(tool_names or ())
        self.reset()

    def reset(self) -> None:
        self._state = OUTSIDE
        self._pending = ""
        self._cur: Optional[Dict[str, Any]] = None   # call under construction
        self._cur_index = -1
        self._calls_seen = 0                         # calls started (channels opened)
        self._calls: List[Dict[str, Any]] = []       # completed calls, OpenAI shape
        self._param_index = 0
        self._recipient = ""  # recipient of the channel currently being parsed
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
            self._finish_call(None)
        elif self._state != OUTSIDE:
            self.errors.append(f"generation ended in state {self._state}")

    def begin_call(self, recipient: str, msg: Optional[dict] = None) -> None:
        """A tool channel opened: start a call named after the recipient.

        The payload's <atem:invoke name="..."> attribute normally matches the
        recipient; a mismatch corrects the streamed name (see _set_invoke_name).
        """
        if self._stopped:
            return
        if self._cur is not None:
            self.errors.append("new tool channel opened before previous call finished")
            self._finish_call(msg)
        self._recipient = recipient
        self._start_call(recipient)

    # -- IncrementalParser API ---------------------------------------------------

    def parse(self, msg: dict, delta_text: str, delta_tokens=None) -> str:
        """Consume one decoded delta. Returns content text for this delta.

        Mutates `msg` with completed calls under msg["tool_calls"][str(index)]
        and emits streaming fragments through `on_fragment`. The driver
        contract is that deltas either carry text (no protocol ids) or
        protocol ids (empty text); if both arrive, text is processed first.
        delta_tokens carries channel-end ids (EOM/EOT/EOS) forwarded by the
        channel splitter while a tool channel is open.
        """
        if self._stopped:
            return ""
        out: List[str] = []
        if delta_text:
            self._pending += delta_text
            self._machine(out, final=bool(delta_tokens), msg=msg)
        for token in (delta_tokens or []):
            tid = int(token)
            if tid in CHANNEL_END_IDS:
                self._machine(out, final=True, msg=msg)
                self._channel_end(msg)
                if self._stopped:
                    self._pending = ""
                    break
        return "".join(out)

    # -- state machine ------------------------------------------------------------

    def _machine(self, out: List[str], final: bool, msg: Optional[dict] = None) -> None:
        """Advance the machine over `self._pending`.

        final=True means the segment ends at a boundary: nothing may be held
        back waiting for more text.
        """
        while True:
            if self._skip_rest:
                self._pending = ""
                return
            state = self._state
            if state == OUTSIDE:
                idx, tag = _find_any(self._pending, (FC_OPEN, INVOKE_OPEN, FC_CLOSE))
                if idx == -1:
                    if final:
                        text, self._pending = self._pending, ""
                        self._emit_outside(out, text)
                        return
                    hold = _holdback_suffix(self._pending, HOLD_TAGS)
                    keep = len(self._pending) - hold
                    if keep:
                        text = self._pending[:keep]
                        self._pending = self._pending[keep:]
                        self._emit_outside(out, text)
                    return
                pre, self._pending = (
                    self._pending[:idx],
                    self._pending[idx + len(tag):],
                )
                self._emit_outside(out, pre)
                if tag == FC_OPEN:
                    self._state = BLOCK
                elif tag == INVOKE_OPEN:
                    # tolerate a missing block wrapper
                    self._ensure_open_call()
                    self._state = INVOKE_NAME
                # else FC_CLOSE: block wrapper after a completed call -> ignore
                continue
            if state == BLOCK:
                idx, tag = _find_any(self._pending, (INVOKE_OPEN, FC_CLOSE))
                if idx == -1:
                    if final:
                        if self._pending.strip():
                            self.errors.append(
                                f"unexpected text in tool block: {self._pending[:20]!r}"
                            )
                        self._pending = ""
                        return
                    hold = _holdback_suffix(self._pending, (INVOKE_OPEN, FC_CLOSE))
                    keep = len(self._pending) - hold
                    if keep:
                        pre = self._pending[:keep]
                        self._pending = self._pending[keep:]
                        if pre.strip():
                            self.errors.append(
                                f"unexpected text in tool block: {pre[:20]!r}"
                            )
                    return
                pre, self._pending = (
                    self._pending[:idx],
                    self._pending[idx + len(tag):],
                )
                if pre.strip():
                    self.errors.append(f"unexpected text in tool block: {pre[:20]!r}")
                if tag == INVOKE_OPEN:
                    self._ensure_open_call()
                    self._state = INVOKE_NAME
                else:  # FC_CLOSE: block closed without an invoke
                    self._state = OUTSIDE
                continue
            if state == INVOKE_NAME:
                end = self._pending.find(ATTR_END)
                if end == -1:
                    if final:
                        self.errors.append(
                            f"malformed invoke name: {self._pending[:20]!r}"
                        )
                        self._skip_rest = True
                        self._pending = ""
                    return
                name, self._pending = (
                    self._pending[:end].strip(),
                    self._pending[end + len(ATTR_END):],
                )
                self._set_invoke_name(name)
                self._state = BODY
                continue
            if state == BODY:
                idx, tag = _find_any(self._pending, (PARAM_OPEN, INVOKE_CLOSE))
                if idx == -1:
                    if final:
                        if self._pending.strip():
                            self.errors.append(
                                f"unexpected text in invoke: {self._pending[:20]!r}"
                            )
                        self._pending = ""
                        return
                    hold = _holdback_suffix(self._pending, (PARAM_OPEN, INVOKE_CLOSE))
                    keep = len(self._pending) - hold
                    if keep:
                        pre = self._pending[:keep]
                        self._pending = self._pending[keep:]
                        if pre.strip():
                            self.errors.append(
                                f"unexpected text in invoke: {pre[:20]!r}"
                            )
                    return
                pre, self._pending = (
                    self._pending[:idx],
                    self._pending[idx + len(tag):],
                )
                if pre.strip():
                    self.errors.append(f"unexpected text in invoke: {pre[:20]!r}")
                if tag == PARAM_OPEN:
                    self._state = PARAM_NAME
                else:  # INVOKE_CLOSE: payload complete
                    self._finish_call(msg)
                continue
            if state == PARAM_NAME:
                end = self._pending.find(ATTR_END)
                if end == -1:
                    if final:
                        self.errors.append(
                            f"malformed parameter name: {self._pending[:20]!r}"
                        )
                        self._skip_rest = True
                        self._pending = ""
                    return
                key, self._pending = (
                    self._pending[:end].strip(),
                    self._pending[end + len(ATTR_END):],
                )
                self._begin_param(key)
                self._state = VALUE
                continue
            if state == VALUE:
                idx = self._pending.find(PARAM_CLOSE)
                if idx == -1:
                    if final:
                        value, self._pending = self._pending, ""
                        self._consume_value(value)
                        self._end_value()
                        self._state = BODY
                        continue
                    hold = _holdback_suffix(self._pending, (PARAM_CLOSE,))
                    keep = len(self._pending) - hold
                    if keep:
                        self._consume_value(self._pending[:keep])
                        self._pending = self._pending[keep:]
                    return
                value, self._pending = (
                    self._pending[:idx],
                    self._pending[idx + len(PARAM_CLOSE):],
                )
                self._consume_value(value)
                self._end_value()
                self._state = BODY
                continue
            return

    # -- boundaries / helpers -------------------------------------------------------

    def _channel_end(self, msg: Optional[dict]) -> None:
        """A control message ended (EOM/EOT/EOS): close any unterminated call.

        A completed call already closed at '</atem:invoke>' (cur is None) is a
        no-op; anything else is an unterminated call -> error + force-close.
        """
        if self._cur is None:
            return
        self.errors.append(
            f"tool channel ended mid-payload (state {self._state})"
        )
        self._finish_call(msg)

    def _emit_outside(self, out: List[str], text: str) -> None:
        if not text:
            return
        if self._calls_seen == 0:
            out.append(text)  # pre-block text in the tool channel -> content
        elif text.strip():
            # Real text after a completed call means the model is rambling
            # past its answer -> stop generation.
            self.status = StreamingStatus.TOOL_CALL_STOP
            self.set_status(self.status)
        # else: whitespace between/after calls -> drop

    def _ensure_open_call(self) -> None:
        """Open a call for a follow-up <atem:invoke> in the same channel.

        begin_call() already opened one for the channel's first invoke; the
        call closes at '</atem:invoke>', so the next invoke in the same
        function_calls block (or channel) needs a fresh call named after the
        recipient until the payload's name attribute corrects it.
        """
        if self._cur is None:
            self._start_call(self._recipient)

    def _normalize_name(self, name: str) -> str:
        """Map a dotted 'namespace.function' name back to the registered tool.

        The template advertises streaming recipients as 'namespace.*'; a model
        that streams 'tools.get_weather' for a registered 'get_weather' tool
        must surface as 'get_weather'. Unknown names surface verbatim.
        """
        if not self._tool_names or name in self._tool_names or "." not in name:
            return name
        fn = name.rsplit(".", 1)[1]
        return fn if fn in self._tool_names else name

    def _start_call(self, name: str = "") -> None:
        name = self._normalize_name(name)
        self._cur_index = self._calls_seen
        self._calls_seen += 1
        self._cur = {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": ""},
        }
        self._param_index = 0
        self._state = OUTSIDE
        self._skip_rest = False
        self._frag({
            "index": self._cur_index,
            "id": self._cur["id"],
            "type": "function",
            "function": {"name": name, "arguments": ""},
        })

    def _set_invoke_name(self, name: str) -> None:
        """Apply the payload's <atem:invoke name="..."> attribute.

        The recipient already named the call; only a disagreement emits a
        correction fragment (clients overwrite the name for the same index).
        """
        if not name:
            self.errors.append("empty invoke name")
            return
        name = self._normalize_name(name)
        current = self._cur["function"]["name"] if self._cur is not None else ""
        if current and current != name:
            self.errors.append(
                f"invoke name {name!r} != recipient {current!r}"
            )
        if self._cur is not None and current != name:
            # fill an empty recipient name or correct a disagreeing one
            self._cur["function"]["name"] = name
            self._frag({"index": self._cur_index, "function": {"name": name}})

    def _begin_param(self, key: str) -> None:
        self._param_index += 1
        sep = "{" if self._param_index == 1 else ", "
        self._frag({
            "index": self._cur_index,
            "function": {"arguments": f"{sep}{json.dumps(key)}: "},
        })
        self._frag({"index": self._cur_index, "function": {"arguments": '"'}})

    def _end_value(self) -> None:
        self._frag({"index": self._cur_index, "function": {"arguments": '"'}})

    def _consume_value(self, text: str) -> None:
        # Values are verbatim payload text (the format spec does not strip
        # whitespace, and the template's own multi-line example value ends
        # with a literal newline that must survive).
        if text:
            self._frag({
                "index": self._cur_index,
                "function": {"arguments": _json_escape_fragment(text)},
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


class MuseGlimmerChannelSplitter:
    """Routes 'to=' channel payloads by token ID (Harmony message framing).

    The stream starts in 'header' (the generation prompt ends with
    '<|start|>assistant'). Header text accumulates until MESSAGE_ID resolves
    the recipient:

        'self'        -> 'reasoning': text streams out as reasoning deltas with
                         a one-line tail held back so the newline before the
                         message end is trimmed.
        None / 'user' -> 'content': text passes through verbatim.
        other         -> 'tool': text and forwarded channel-end ids stream into
                         the tool parser; begin_call(recipient) is issued at
                         channel open.

    START_ID implicitly closes an open channel (reasoning tail flushed, open
    tool call force-closed by forwarding EOM_ID) and reopens header state.
    CHANNEL_END_IDS close a channel: reasoning tail flushed, tool parser
    finalized via the forwarded id. After a channel closes, state falls to
    'content' so stray text between messages is never silently dropped.

    The driver contract is exact attribution: each feed() call either carries
    text (no protocol ids) or protocol ids (empty text).
    """

    def __init__(self, tool_parser: Optional[MuseGlimmerToolParser] = None):
        self.tool_parser = tool_parser
        self._state = "header"  # header | reasoning | content | tool
        self._header = ""
        self._held = ""  # held reasoning tail (one line)

    def feed(self, text: str, token_ids) -> Tuple[str, str, str, List[int]]:
        """Returns (reasoning_delta, content_delta, tool_text, tool_ids)."""
        reason_parts: List[str] = []
        content_parts: List[str] = []
        tool_parts: List[str] = []
        tool_ids: List[int] = []
        for token in (token_ids or []):
            tid = int(token)
            if tid == START_ID:
                # A new message opens; implicitly close any open channel.
                if self._state == "tool":
                    tool_ids.append(EOM_ID)  # force-close the dangling call
                elif self._state == "reasoning":
                    tail = self._held.rstrip("\n")
                    if tail:
                        reason_parts.append(tail)
                    self._held = ""
                self._state = "header"
                self._header = ""
            elif tid == MESSAGE_ID:
                if self._state == "header":
                    self._open_channel(reason_parts, content_parts)
                else:
                    logger.debug("museglimmer <|message|> outside header state")
            elif tid in CHANNEL_END_IDS:
                self._close_channel(reason_parts, tool_ids, tid)
            # unknown control ids ignored
        if text:
            if self._state == "header":
                self._header += text
                if len(self._header) > HEADER_LIMIT:
                    logger.debug("museglimmer header never resolved: %r", self._header[:40])
                    content_parts.append(self._header)
                    self._header = ""
                    self._state = "content"
            elif self._state == "reasoning":
                # Stream reasoning out, keeping a one-line tail so the
                # newline before the message end can be trimmed cleanly.
                self._held += text
                cut = self._held.rfind("\n")
                if cut > 0:
                    reason_parts.append(self._held[:cut])
                    self._held = self._held[cut:]
            elif self._state == "tool":
                tool_parts.append(text)
            else:  # content (or stray text after a close)
                content_parts.append(text)
        return (
            "".join(reason_parts),
            "".join(content_parts),
            "".join(tool_parts),
            tool_ids,
        )

    def finalize(self) -> Tuple[str, str]:
        """Flush held text at end of generation. Returns (reasoning, content)."""
        reason = content = ""
        if self._state == "reasoning":
            tail = self._held.rstrip("\n")
            if tail:
                reason = tail
            self._held = ""
        elif self._state == "header" and self._header:
            # Header never resolved (<|message|> never arrived): surface it as
            # content rather than dropping the model's text.
            content = self._header
            self._header = ""
        self._state = "content"
        return reason, content

    def _open_channel(self, reason_parts, content_parts) -> None:
        recipient_m = RECIPIENT_RE.search(self._header)
        recipient = recipient_m.group(1) if recipient_m else None
        self._header = ""
        if recipient == SELF:
            self._state = "reasoning"
            self._held = ""
        elif recipient in (None, USER):
            self._state = "content"
        else:
            self._state = "tool"
            if self.tool_parser is not None:
                self.tool_parser.begin_call(recipient)

    def _close_channel(self, reason_parts, tool_ids, boundary_id: int) -> None:
        if self._state == "reasoning":
            tail = self._held.rstrip("\n")
            if tail:
                reason_parts.append(tail)
            self._held = ""
        elif self._state == "tool":
            tool_ids.append(boundary_id)  # finalize the call via the parser
        self._state = "content"


class MuseGlimmerToolCallStreamer(StreamerBase):
    """Engine streamer for museglimmer tool requests.

    StreamerBase with its own incremental decode (cumulative decode + delta
    slicing, like ChunkStreamer): Harmony control tokens are matched by ID and
    fed through the channel splitter / tool parser as ID-only events, while
    everything else is decoded incrementally and fed as token-less text
    deltas. special=True tokens vanish from decode(), which is exactly why
    they must be intercepted here. This deliberately avoids TextParserStreamer,
    whose parser chain deadlocks when generation runs on a worker thread
    under asyncio.

    Parsed OpenAI deltas are enqueued on text_queue (same contract as
    ChunkStreamer) wrapped as {"chat_delta": [...]} so the route can
    distinguish them from metrics/error dicts.
    """

    def __init__(self, tokenizer, gen_config):
        super().__init__()
        tool_names = {
            t["function"]["name"]
            for t in (getattr(gen_config, "tools", None) or ())
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
            and t["function"].get("name")
        }
        self.tool_parser = MuseGlimmerToolParser(tool_names=tool_names or None)
        self.tool_parser.on_fragment = self._collect_fragment
        self._fragments: List[Dict[str, Any]] = []
        self._channels = MuseGlimmerChannelSplitter(self.tool_parser)
        # Raw tagged-output reconstruction: decoded text deltas plus the tag
        # text of every intercepted control id. Lets non-streaming callers
        # obtain the raw text parse_generation expects without relying on
        # pipeline-level skip_special_tokens (not available on this wheel).
        self._raw_parts: List[str] = []
        self.tokenizer = tokenizer
        self._protocol_tags: Dict[int, str] = {}
        for tag, tag_id in (
            (START, START_ID),
            (MESSAGE, MESSAGE_ID),
            (EOM, EOM_ID),
            (EOT, EOT_ID),
            (EOS, EOS_ID),
        ):
            ids = tokenizer.encode(tag).input_ids.data.tolist()[0]
            # openvino_genai Tokenizer.encode() prepends the model's BOS token
            # on this export; strip it so the tag's own single id remains.
            if len(ids) == 2 and ids[0] == BOS_ID:
                ids = ids[1:]
            if len(ids) == 1:
                if ids[0] != tag_id:
                    logger.warning(
                        "museglimmer tag %r encodes to id %d, expected %d",
                        tag, ids[0], tag_id,
                    )
                self._protocol_tags[ids[0]] = tag
            else:
                logger.warning(
                    "museglimmer tag %r is not a single token (encodes to %s); "
                    "it will not be intercepted",
                    tag, ids,
                )
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
        reason_delta, content_delta, tool_text, tool_ids = self._channels.feed(
            text, delta_tokens
        )
        parsed_content = self.tool_parser.parse({}, tool_text, tool_ids)
        deltas: List[Dict[str, Any]] = []
        if reason_delta:
            deltas.append({"reasoning_content": reason_delta})
        content = content_delta + parsed_content
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

        Called before an intercepted control token: pre-boundary text must be
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
            tag = self._protocol_tags.get(int(tid))
            if tag is not None:
                self._flush_text()
                self._raw_parts.append(tag)
                self._process_delta("", [int(tid)])
            else:
                self.tokens_cache.append(int(tid))
                self._decode_available()
        return self.tool_parser.status

    @property
    def raw_text(self) -> str:
        """Reconstructed raw tagged output (control tokens as literal text)."""
        return "".join(self._raw_parts)

    def end(self) -> None:
        self._flush_text()
        reason, content = self._channels.finalize()
        deltas: List[Dict[str, Any]] = []
        if reason:
            deltas.append({"reasoning_content": reason})
        if content:
            deltas.append({"content": content})
        if deltas:
            self._enqueue({"chat_delta": deltas})
        self.tool_parser.finalize()
        if self.tool_parser.errors:
            logger.warning("museglimmer tool parser errors: %s", self.tool_parser.errors)
        self._enqueue(None)

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()


# -- raw-text path (non-streaming) ------------------------------------------------


def parse_generation(
    text: str,
    tools: Optional[List[Dict[str, Any]]] = None,  # accepted; unused
    enable_thinking: bool = True,  # accepted for dispatch symmetry; unused
) -> tuple[str, str, Optional[List[Dict[str, Any]]]]:
    """Split raw tagged model output into (reasoning, content, tool_calls).

    Expects text with literal control tokens (e.g.
    MuseGlimmerToolCallStreamer.raw_text; the VLM decode always strips
    specials, so pipeline-level decode output cannot be parsed). Returns
    ("", text, None) when no control tokens are present at all.
    """
    if not text:
        return "", "", None
    if MESSAGE not in text and EOM not in text and EOT not in text and EOS not in text:
        return "", text, None
    if not text.startswith(START):
        # The first message's header arrives without <|start|> (the generation
        # prompt already ends with '<|start|>assistant').
        text = START + text
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    calls: List[Dict[str, Any]] = []
    for segment in text.split(START):
        if not segment:
            continue
        end_idx, end_tag = _find_any(segment, (EOM, EOT, EOS))
        body = segment[:end_idx] if end_idx != -1 else segment
        trailer = (
            segment[end_idx + len(end_tag):] if end_idx != -1 else ""
        )
        if trailer.strip():
            logger.debug(
                "museglimmer stray text after message end: %r", trailer[:40]
            )
        header, sep, payload = body.partition(MESSAGE)
        if not sep:
            # No message token: not a framed message; surface as content.
            content_parts.append(header)
            continue
        recipient_m = RECIPIENT_RE.search(header)
        recipient = recipient_m.group(1) if recipient_m else None
        if recipient == SELF:
            reasoning_parts.append(payload.rstrip("\n"))
        elif recipient in (None, USER):
            content_parts.append(payload)
        else:
            parser = MuseGlimmerToolParser(
                tool_names={t["function"]["name"] for t in (tools or [])
                            if isinstance(t, dict) and isinstance(t.get("function"), dict)
                            and t["function"].get("name")} or None
            )
            parser.begin_call(recipient)
            leftover = parser.parse({}, payload, [EOM_ID])
            parser.finalize()
            if parser.errors:
                logger.debug("museglimmer tool parser errors: %s", parser.errors)
            if leftover.strip():
                content_parts.append(leftover)
            calls.extend(parser.completed_calls)
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts).strip()
    return reasoning, content, calls or None


if __name__ == "__main__":
    # Live smoke test for the engine path (MuseGlimmerToolCallStreamer):
    # reasoning + parallel tool calls, then a plain no-tools answer.
    #   python -m src.engine.ov_genai.tool_parse.museglimmer [DEVICE] [MODEL_PATH] [CACHE_DIR]
    import sys

    import openvino_genai as ov

    DEVICE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "GPU.0"
    MODEL_PATH = (
        sys.argv[2] if len(sys.argv) > 2 else
        "/mnt/Ironwolf-4TB/Models/OpenVINO/Muse-Glimmer-30B-int4-ov"
    )
    CACHE_DIR = sys.argv[3] if len(sys.argv) > 3 else "/home/echo/.cache/openvino"
    SMOKE_TOOLS = [
        {
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
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current local time for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        },
    ]

    pipe = ov.VLMPipeline(MODEL_PATH, DEVICE, CACHE_DIR=CACHE_DIR)
    tokenizer = pipe.get_tokenizer()

    def run(tools, question, thinking=True, max_new_tokens=1024):
        from types import SimpleNamespace

        gen_config = SimpleNamespace(
            tools=tools, chat_template_kwargs={"enable_thinking": thinking}
        )
        streamer = MuseGlimmerToolCallStreamer(tokenizer, gen_config)
        history = ov.ChatHistory([{"role": "user", "content": question}])
        if tools:
            history.set_tools(tools)
        config = ov.GenerationConfig()
        config.max_new_tokens = max_new_tokens
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

        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        tool_frags = [f for d in deltas for f in d.get("tool_calls", [])]
        print("=" * 60)
        print("reasoning:", repr(reasoning[:300]))
        print("content:", repr(content[:300]))
        names = [f["function"]["name"] for f in tool_frags if f.get("function", {}).get("name")]
        print("names:", names)
        per_call: Dict[str, str] = {}
        for f in tool_frags:
            fn = f.get("function", {})
            if "arguments" in fn:
                per_call[str(f["index"])] = per_call.get(str(f["index"]), "") + fn["arguments"]
        ok = True
        for index, args in sorted(per_call.items()):
            try:
                print(f"call {index} args:", json.loads(args))
            except json.JSONDecodeError as exc:
                ok = False
                print(f"call {index} args INVALID: {args!r} ({exc})")
        print("parser errors:", streamer.tool_parser.errors)
        print("parser status:", streamer.tool_parser.get_status())
        if reasoning and names and ok:
            print("REASONING + TOOLS OK")
        elif not tools and content and not names and ok:
            print("PLAIN CONTENT OK")
        print("RAW TEXT:")
        print(streamer.raw_text)
        print("=" * 60)
        return reasoning, content, names, per_call

    run(
        SMOKE_TOOLS,
        "You must call BOTH tools before answering: first get_weather for "
        "Tokyo, then get_time for Paris. Make both calls now in this turn.",
        max_new_tokens=2048,
    )
    run(
        None,
        "Say hello and nothing else.",
        thinking=False,
        max_new_tokens=256,
    )
