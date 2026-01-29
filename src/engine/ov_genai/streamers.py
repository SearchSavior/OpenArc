from typing import List, Optional, Union
import openvino_genai
import asyncio

from openvino_genai import StreamerBase
from src.server.models.ov_genai import OVGenAI_GenConfig


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
        self._cancelled = asyncio.Event()          # cancellation flag for thread-safe signaling

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        # Check for cancellation first
        if self._cancelled.is_set():
            # Signal completion to the queue so the consumer can exit
            self.text_queue.put_nowait(None)
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
                if chunk:
                    self.text_queue.put_nowait(chunk)
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
                self.text_queue.put_nowait(chunk)
        # Signal completion
        self.text_queue.put_nowait(None)


class BlockStreamer(StreamerBase):
    """
    Non-streaming (block) mode streamer.
    Collects all tokens during generation and emits the complete text as a single block
    when generation ends. Used for stream=False mode.

    Unlike ChunkStreamer, this does not emit partial results during generation -
    the entire response is yielded at once.
    """
    def __init__(self, decoder_tokenizer):
        super().__init__()
        self.decoder_tokenizer = decoder_tokenizer
        self.tokens_cache: List[int] = []
        self.text_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        self._cancelled = asyncio.Event()

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
        # Check for cancellation first
        if self._cancelled.is_set():
            self.text_queue.put_nowait(None)
            return openvino_genai.StreamingStatus.CANCEL

        # Collect tokens without emitting
        if isinstance(token, list):
            self.tokens_cache.extend(token)
        else:
            self.tokens_cache.append(token)

        return openvino_genai.StreamingStatus.RUNNING

    def cancel(self) -> None:
        """Signal cancellation of the generation."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been signaled."""
        return self._cancelled.is_set()

    def end(self) -> None:
        # Decode and emit all tokens as a single block
        text = self.decoder_tokenizer.decode(self.tokens_cache)
        if text:
            self.text_queue.put_nowait(text)
        # Signal completion
        self.text_queue.put_nowait(None)