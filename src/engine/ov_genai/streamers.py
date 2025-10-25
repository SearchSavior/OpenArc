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

    def write(self, token: Union[int, List[int]]) -> openvino_genai.StreamingStatus:
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

    def end(self) -> None:
        # Flush any remaining tokens at the end
        text = self.decoder_tokenizer.decode(self.tokens_cache)
        if len(text) > self.last_print_len:
            chunk = text[self.last_print_len:]
            if chunk:
                self.text_queue.put_nowait(chunk)
        # Signal completion
        self.text_queue.put_nowait(None)
