from typing import List, Optional, Union
import openvino_genai
import asyncio

from openvino_genai import StreamerBase
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.engine.ov_genai.tool_parse import gemma4 as gemma4_tool_parse
from src.engine.ov_genai.tool_parse import qwen35 as qwen35_tool_parse


class ChunkStreamer(StreamerBase):
    """
    Streams decoded text in chunks of N tokens.
    - tokens_len == 1 → token-by-token streaming.
    - tokens_len  > 1 → emit after every N tokens.
    Uses cumulative decode + delta slicing to avoid subword boundary artifacts.
    """
    def __init__(self, decoder_tokenizer, gen_config: OVGenAI_GenConfig):
        super().__init__()
        self.decoder_tokenizer = decoder_tokenizer
        self.tokens_len = (gen_config.stream_chunk_tokens)  # enforce at least 1
        self.tokens_cache: List[int] = []          # cumulative token buffer
        self.since_last_emit: int = 0              # tokens collected since last emit
        self.last_print_len: int = 0               # length of decoded text we've already emitted
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

        # Normalize input to a list of ints
        if isinstance(token, list):
            self.tokens_cache.extend(token)
            self.since_last_emit += len(token)
        else:
            self.tokens_cache.append(token)
            self.since_last_emit += 1

        # Only emit when we've reached the chunk boundary
        if self.since_last_emit >= self.tokens_len:
            text = self.decoder_tokenizer.decode(self.tokens_cache)
            # Emit only the newly materialized portion
            if len(text) > self.last_print_len:
                chunk = text[self.last_print_len:]
                if chr(65533) in chunk:
                    self.since_last_emit -= 1
                    return openvino_genai.StreamingStatus.RUNNING
                if chunk:
                    self._enqueue(chunk)
                self.last_print_len = len(text)
            self.since_last_emit = 0

        return openvino_genai.StreamingStatus.RUNNING

    def cancel(self) -> None:
        """Signal cancellation of the streaming generation."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been signaled."""
        return self._cancelled.is_set()

    def end(self) -> None:
        # Flush any remaining tokens at the end
        text = self.decoder_tokenizer.decode(self.tokens_cache)
        if len(text) > self.last_print_len:
            chunk = text[self.last_print_len:]
            if chunk:
                self._enqueue(chunk)
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
    split reasoning); everything else uses ChunkStreamer. All of them enqueue
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
    return ChunkStreamer(tokenizer, gen_config)