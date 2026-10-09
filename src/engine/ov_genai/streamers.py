from typing import List, Optional, Union
import openvino_genai
import asyncio

from openvino_genai import StreamerBase
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.engine.ov_genai.tool_parse import gemma4 as gemma4_tool_parse
from src.engine.ov_genai.tool_parse import museglimmer as museglimmer_tool_parse
from src.engine.ov_genai.tool_parse import qwen35 as qwen35_tool_parse


class IncrementalChunkDecoder:
    """
    Incremental detokenizer that emits decoded text in chunks of N tokens.
    - tokens_len == 1 → token-by-token emission.
    - tokens_len  > 1 → emit after every N tokens.

    Only the tokens since the last emit are decoded (BPE decode is
    concatenative, so decoding the pending segment yields exactly the suffix a
    full re-decode would produce). A decoded segment containing U+FFFD is held
    back until more tokens complete the character, matching the old full-cache
    behavior. Cost per emit is constant instead of growing with the square of
    the output length.

    Plain helper shared by ChunkStreamer and the continuous-batching engine;
    it owns no queues or callbacks.
    """
    def __init__(self, decoder_tokenizer, tokens_len: int):
        self.decoder_tokenizer = decoder_tokenizer
        self.tokens_len = tokens_len  # enforce at least 1
        self.tokens_pending: List[int] = []      # tokens not yet emitted as text
        self.since_last_emit: int = 0            # tokens collected since last emit

    def feed(self, token_ids: Union[int, List[int]]) -> Optional[str]:
        """Buffer token ids; return decoded text at a chunk boundary, else None.

        None is also returned when the decoded segment holds a partial UTF-8
        character (U+FFFD): the pending buffer keeps every unemitted token, so
        the next attempt decodes the same suffix the old full-cache decode
        would.
        """
        # Normalize input to a list of ints
        if isinstance(token_ids, list):
            self.tokens_pending.extend(token_ids)
            self.since_last_emit += len(token_ids)
        else:
            self.tokens_pending.append(token_ids)
            self.since_last_emit += 1

        # Only emit when we've reached the chunk boundary
        if self.since_last_emit >= self.tokens_len:
            text = self.decoder_tokenizer.decode(self.tokens_pending)
            if chr(65533) in text:
                return None
            self.tokens_pending = []
            self.since_last_emit = 0
            return text if text else None
        return None

    def flush(self) -> Optional[str]:
        """Decode and return any tokens still pending at end of generation."""
        if not self.tokens_pending:
            return None
        text = self.decoder_tokenizer.decode(self.tokens_pending)
        self.tokens_pending = []
        self.since_last_emit = 0
        return text if text else None


class ChunkStreamer(StreamerBase):
    """
    Streams decoded text in chunks of N tokens via IncrementalChunkDecoder.
    Enqueues emitted chunks (and the None EOF sentinel) on .text_queue.
    """
    def __init__(self, decoder_tokenizer, gen_config: OVGenAI_GenConfig):
        super().__init__()
        self._decoder = IncrementalChunkDecoder(
            decoder_tokenizer, gen_config.stream_chunk_tokens
        )
        self.text_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        self._cancelled = asyncio.Event()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _enqueue(self, msg: Optional[str]) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.text_queue.put_nowait, msg)
        else:
            self.text_queue.put_nowait(msg)

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        # Check for cancellation first
        if self._cancelled.is_set():
            # Signal completion to the queue so the consumer can exit
            self._enqueue(None)
            return openvino_genai.StreamingStatus.CANCEL

        text = self._decoder.feed(token)
        if text:
            self._enqueue(text)

        return openvino_genai.StreamingStatus.RUNNING

    def cancel(self) -> None:
        """Signal cancellation of the streaming generation."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been signaled."""
        return self._cancelled.is_set()

    def end(self) -> None:
        # Flush any remaining tokens at the end
        text = self._decoder.flush()
        if text:
            self._enqueue(text)
        self._enqueue(None)


def ensure_tool_call_parser(gen_config: OVGenAI_GenConfig, load_config) -> None:
    """Fall back to the model's registered tool-call parser when the request
    does not carry one (routes set gen_config.tool_call_parser; direct engine
    callers such as tests and bench rely on the load-time registration)."""
    if getattr(gen_config, "tool_call_parser", None) is None:
        parser = getattr(load_config, "tool_call_parser", None)
        if parser is not None:
            gen_config.tool_call_parser = parser.value


def select_streamer(tokenizer, gen_config: OVGenAI_GenConfig) -> StreamerBase:
    """Pick the streaming implementation for a generation request.

    Tool-call requests on qwen35 models stream through Qwen35ToolCallStreamer
    (token-ID block boundaries, parsed OpenAI deltas). gemma4 requests use
    Gemma4ToolCallStreamer whenever tools are requested OR thinking is enabled
    (its thought-channel tags are special=True, so the plain text path cannot
    split reasoning). museglimmer requests ALWAYS use MuseGlimmerToolCallStreamer:
    its Harmony 'to=' channel routing is token-ID-only, so even a plain content
    response opens with ' to=user<|message|>' header text that only the engine
    streamer can strip. Everything else uses ChunkStreamer. All of them enqueue
    on .text_queue, so consumers are unaffected.
    """
    parser_name = getattr(gen_config, "tool_call_parser", None)
    if gen_config.tools and parser_name == "qwen35":
        return qwen35_tool_parse.Qwen35ToolCallStreamer(tokenizer, gen_config)
    if parser_name == "gemma4":
        thinking = True
        if getattr(gen_config, "chat_template_kwargs", None):
            thinking = bool(gen_config.chat_template_kwargs.get("enable_thinking", True))
        if gen_config.tools or thinking:
            return gemma4_tool_parse.Gemma4ToolCallStreamer(tokenizer, gen_config)
    if parser_name == "museglimmer":
        return museglimmer_tool_parse.MuseGlimmerToolCallStreamer(tokenizer, gen_config)
    return ChunkStreamer(tokenizer, gen_config)