#!/usr/bin/env python3
"""
OpenVINO inference script for Qwen3-ASR (0.6B / 1.7B).

Runs inference from OpenVINO IR models produced by qwen3_asr_ov_convert.py:
  - audio_encoder.xml/.bin
  - thinker_embeddings.xml/.bin
  - decoder.xml/.bin

Usage:
    pip install torch openvino numpy soundfile
    python qwen3_asr_ov_infer.py <audio.wav> [--ov-dir ov_model] [--device CPU]
"""

import os
import json
import argparse
import asyncio
import time
import base64
from pathlib import Path
import logging
import gc
from typing import Any, AsyncIterator, Dict, Optional, Union

import numpy as np
import openvino as ov
import torch
from pydantic import BaseModel, Field, field_validator

from src.engine.openvino.qwen3_asr.asr_utils import (
    MAX_ASR_INPUT_SECONDS,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)
from src.server.model_registry import ModelRegistry
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400

# Token IDs

TOKEN_IM_START = 151644
TOKEN_IM_END = 151645
TOKEN_AUDIO_START = 151669
TOKEN_AUDIO_END = 151670
TOKEN_AUDIO_PAD = 151676
TOKEN_ENDOFTEXT = 151643
TOKEN_ASR_TEXT = 151704
EOS_TOKEN_IDS = {TOKEN_ENDOFTEXT, TOKEN_IM_END}
PROMPT_PREFIX = [
    TOKEN_IM_START, 8948, 198, TOKEN_IM_END, 198,
    TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START,
]
PROMPT_SUFFIX = [
    TOKEN_AUDIO_END, TOKEN_IM_END, 198,
    TOKEN_IM_START, 77091, 198,
]


class Qwen3ASRConfig(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio payload")
    language: Optional[str] = Field(default=None, description="Optional forced language")
    max_tokens: int = Field(default=1024, description="Maximum generated tokens per chunk")
    max_chunk_sec: float = Field(default=30.0, description="Chunk size upper bound in seconds")
    search_expand_sec: float = Field(default=5.0, description="Boundary search expansion in seconds")
    min_window_ms: float = Field(default=100.0, description="Energy window in milliseconds")

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("max_chunk_sec", "search_expand_sec", "min_window_ms")
    @classmethod
    def _validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("numeric values must be positive")
        return v


class Qwen3ASRHelpers:
    @staticmethod
    def hf_config(config_path: Path) -> dict:
        with open(config_path) as f:
            cfg = json.load(f)
        if "dec_layers" in cfg:
            return cfg

        tc = cfg["thinker_config"]
        ac = tc["audio_config"]
        txc = tc["text_config"]
        return {
            "enc_n_window": ac["n_window"],
            "dec_layers": txc["num_hidden_layers"],
            "dec_kv_heads": txc["num_key_value_heads"],
            "dec_head_dim": txc["head_dim"],
        }

    @staticmethod
    def hertz_to_mel(freq):
        mels = 3.0 * freq / 200.0
        if isinstance(freq, np.ndarray):
            log_region = freq >= 1000.0
            mels[log_region] = 15.0 + np.log(freq[log_region] / 1000.0) * (27.0 / np.log(6.4))
        elif freq >= 1000.0:
            mels = 15.0 + np.log(freq / 1000.0) * (27.0 / np.log(6.4))
        return mels

    @staticmethod
    def mel_to_hertz(mels):
        freq = 200.0 * mels / 3.0
        log_region = mels >= 15.0
        freq[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mels[log_region] - 15.0))
        return freq

    @staticmethod
    def compute_mel_filters():
        num_freq = 1 + WINDOW_SIZE // 2
        fft_freqs = np.linspace(0, SAMPLE_RATE // 2, num_freq)
        mel_freqs = np.linspace(
            Qwen3ASRHelpers.hertz_to_mel(0.0),
            Qwen3ASRHelpers.hertz_to_mel(8000.0),
            NUM_MEL_BINS + 2,
        )
        filter_freqs = Qwen3ASRHelpers.mel_to_hertz(mel_freqs)
        fdiff = np.diff(filter_freqs)
        slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
        down = -slopes[:, :-2] / fdiff[:-1]
        up = slopes[:, 2:] / fdiff[1:]
        fb = np.maximum(0, np.minimum(down, up))
        enorm = 2.0 / (filter_freqs[2 : NUM_MEL_BINS + 2] - filter_freqs[:NUM_MEL_BINS])
        fb *= enorm[np.newaxis, :]
        return fb.astype(np.float32)

    @staticmethod
    def compute_mel_spectrogram(audio_np, mel_filters_np):
        audio = torch.from_numpy(audio_np).float()
        mel_filters = torch.from_numpy(mel_filters_np).float()
        window = torch.hann_window(WINDOW_SIZE)
        stft = torch.stft(audio, WINDOW_SIZE, HOP_LENGTH, window=window, return_complex=True)
        mag2 = stft[..., :-1].abs() ** 2
        mel_spec = mel_filters.T @ mag2
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.numpy()

    @staticmethod
    def count_encoder_tokens(total_frames: int, chunk_size: int = 100) -> int:
        count = 0
        for start in range(0, total_frames, chunk_size):
            chunk_len = min(chunk_size, total_frames - start)
            t = chunk_len
            for _ in range(3):
                t = (t + 1) // 2
            count += t
        return count

    @staticmethod
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("\xa1"), ord("\xac") + 1)) + \
             list(range(ord("\xae"), ord("\xff") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    @staticmethod
    def decode_tokens(token_ids, tokenizer_dir: str) -> str:
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        id_to_token = {v: k for k, v in vocab.items()}

        special_tokens = set()
        tc_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        if os.path.exists(tc_path):
            with open(tc_path) as f:
                tc = json.load(f)
            for tid_str in tc.get("added_tokens_decoder", {}):
                special_tokens.add(int(tid_str))

        byte_enc = Qwen3ASRHelpers.bytes_to_unicode()
        byte_dec = {v: k for k, v in byte_enc.items()}

        pieces = []
        for tid in token_ids:
            if tid in special_tokens:
                if tid == TOKEN_ASR_TEXT:
                    pieces.append("<asr_text>")
                continue
            tok = id_to_token.get(tid, "")
            if tok:
                pieces.append(tok)

        text = "".join(pieces)
        return bytearray([byte_dec[c] for c in text if c in byte_dec]).decode("utf-8", errors="replace")


class OVQwen3ASR:

    def __init__(self, load_config: ModelLoadConfig):
        self.load_config = load_config
        self.ov_dir = Path(load_config.model_path)
        self.runtime_cfg = Qwen3ASRHelpers.hf_config(self.ov_dir / "config.json")
        self.chunk_size = self.runtime_cfg["enc_n_window"] * 2
        self.mel_filters = Qwen3ASRHelpers.compute_mel_filters()
        self.core = ov.Core()
        self.t_model_load = 0.0
        self.enc_model = None
        self.emb_model = None
        self.dec_model = None
        self.dec_request = None

    def load_model(self, load_config: ModelLoadConfig) -> None:
        self.load_config = load_config
        self.ov_dir = Path(load_config.model_path)
        logger.info(f"[{load_config.model_name}] Loading OpenVINO Qwen3 ASR models")
        t_load_start = time.perf_counter()
        self.enc_model = self.core.compile_model(
            str(self.ov_dir / "audio_encoder_model.xml"),
            load_config.device,
        )
        self.emb_model = self.core.compile_model(
            str(self.ov_dir / "thinker_embeddings_model.xml"),
            load_config.device,
        )
        self.dec_model = self.core.compile_model(
            str(self.ov_dir / "decoder_model.xml"),
            load_config.device,
        )
        self.dec_request = self.dec_model.create_infer_request()
        self.t_model_load = time.perf_counter() - t_load_start

    def _embed_tokens(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]
        out = self.emb_model([ids])
        return out[self.emb_model.output(0)]

    def collect_metrics(
        self,
        *,
        feature_sec: float,
        encoder_sec: float,
        prefill_sec: float,
        decode_sec: float,
        detok_sec: float,
        prompt_tokens: int,
        generated_tokens: int,
        encoder_tokens: int,
    ) -> dict:
        prefill_tok_s = (prompt_tokens / prefill_sec) if prefill_sec > 0 else 0.0
        decode_tok_s = (generated_tokens / decode_sec) if decode_sec > 0 else 0.0
        return {
            "feature_sec": feature_sec,
            "encoder_sec": encoder_sec,
            "prefill_sec": prefill_sec,
            "prefill_tok_s": prefill_tok_s,
            "decode_sec": decode_sec,
            "decode_tok_s": decode_tok_s,
            "detok_sec": detok_sec,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "encoder_tokens": encoder_tokens,
        }

    async def audio_chunks(self, chunk_audio: np.ndarray, max_tokens: int):
        t_feature_start = time.perf_counter()
        mel = Qwen3ASRHelpers.compute_mel_spectrogram(chunk_audio, self.mel_filters)
        t_feature = time.perf_counter() - t_feature_start
        total_frames = mel.shape[1]
        expected_tokens = Qwen3ASRHelpers.count_encoder_tokens(total_frames, self.chunk_size)

        pad_len = (self.chunk_size - total_frames % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            mel = np.pad(mel, ((0, 0), (0, pad_len)))
        mel_input = mel[np.newaxis, :, :].astype(np.float32)

        t_encoder_start = time.perf_counter()
        enc_out = self.enc_model([mel_input])
        t_encoder = time.perf_counter() - t_encoder_start
        audio_embeds = enc_out[self.enc_model.output(0)]
        audio_embeds = audio_embeds[0, :expected_tokens, :]
        n_audio = audio_embeds.shape[0]

        input_ids = PROMPT_PREFIX + [TOKEN_AUDIO_PAD] * n_audio + PROMPT_SUFFIX
        input_embeds = self._embed_tokens(input_ids)[0].copy()
        pad_start = len(PROMPT_PREFIX)
        input_embeds[pad_start:pad_start + n_audio] = audio_embeds
        input_embeds = input_embeds[np.newaxis, :, :].astype(np.float32)

        prompt_len = len(input_ids)
        position_ids = np.arange(prompt_len, dtype=np.int64)[np.newaxis, :]


        t_prefill_start = time.perf_counter()
        self.dec_request.reset_state()
        self.dec_request.set_input_tensor(0, ov.Tensor(input_embeds))
        self.dec_request.set_input_tensor(1, ov.Tensor(position_ids))
        self.dec_request.infer()
        logits = self.dec_request.get_output_tensor(0).data
        t_prefill = time.perf_counter() - t_prefill_start

        token = int(np.argmax(logits[0, 0]))
        generated = [token]

        t_decode_start = time.perf_counter()
        for step in range(max_tokens - 1):
            if token in EOS_TOKEN_IDS:
                break
            pos = prompt_len + step
            embed = self._embed_tokens([token]).astype(np.float32)
            pos_id = np.array([[pos]], dtype=np.int64)

            self.dec_request.set_input_tensor(0, ov.Tensor(embed))
            self.dec_request.set_input_tensor(1, ov.Tensor(pos_id))
            self.dec_request.infer()
            logits = self.dec_request.get_output_tensor(0).data

            token = int(np.argmax(logits[0, 0]))
            generated.append(token)
        t_decode = time.perf_counter() - t_decode_start

        while generated and generated[-1] in EOS_TOKEN_IDS:
            generated.pop()

        t_detok_start = time.perf_counter()
        raw = Qwen3ASRHelpers.decode_tokens(generated, str(self.ov_dir))
        t_detok = time.perf_counter() - t_detok_start

        metrics = self.collect_metrics(
            feature_sec=t_feature,
            encoder_sec=t_encoder,
            prefill_sec=t_prefill,
            decode_sec=t_decode,
            detok_sec=t_detok,
            prompt_tokens=prompt_len,
            generated_tokens=len(generated),
            encoder_tokens=expected_tokens,
        )
        return raw, metrics

    async def transcribe(self, gen_config: Qwen3ASRConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        t_transcribe_start = time.perf_counter()
        audio_input = gen_config.audio_base64
        if not audio_input.startswith("data:audio"):
            audio_input = f"data:audio/wav;base64,{audio_input}"
        audio_array = (await asyncio.to_thread(normalize_audios, audio_input))[0]
        language: Optional[str] = None
        if gen_config.language:
            language = normalize_language_name(gen_config.language)
            validate_language(language)

        audio_seconds = len(audio_array) / SAMPLE_RATE
        if audio_seconds <= 0:
            yield {}
            yield ""
            return

        max_chunk_sec = min(float(gen_config.max_chunk_sec), float(MAX_ASR_INPUT_SECONDS))
        chunk_items = await asyncio.to_thread(
            split_audio_into_chunks,
            wav=audio_array,
            sr=SAMPLE_RATE,
            max_chunk_sec=max_chunk_sec,
            search_expand_sec=gen_config.search_expand_sec,
            min_window_ms=gen_config.min_window_ms,
        )
        logger.info(f"[{self.load_config.model_name}] Running {len(chunk_items)} chunk(s)")

        langs = []
        texts = []
        agg = {
            "feature_sec": 0.0,
            "encoder_sec": 0.0,
            "prefill_sec": 0.0,
            "decode_sec": 0.0,
            "detok_sec": 0.0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "encoder_tokens": 0,
        }
        for idx, (chunk_wav, chunk_offset_sec) in enumerate(chunk_items):
            chunk_sec = len(chunk_wav) / SAMPLE_RATE
            logger.info(
                f"[{self.load_config.model_name}] Chunk {idx + 1}/{len(chunk_items)} "
                f"offset={chunk_offset_sec:.2f}s duration={chunk_sec:.2f}s"
            )
            raw, chunk_metrics = await self.audio_chunks(chunk_wav, gen_config.max_tokens)
            lang, text = parse_asr_output(raw, language=language)
            langs.append(lang)
            if text:
                texts.append(text)
            agg["feature_sec"] += chunk_metrics["feature_sec"]
            agg["encoder_sec"] += chunk_metrics["encoder_sec"]
            agg["prefill_sec"] += chunk_metrics["prefill_sec"]
            agg["decode_sec"] += chunk_metrics["decode_sec"]
            agg["detok_sec"] += chunk_metrics["detok_sec"]
            agg["prompt_tokens"] += chunk_metrics["prompt_tokens"]
            agg["generated_tokens"] += chunk_metrics["generated_tokens"]
            agg["encoder_tokens"] += chunk_metrics["encoder_tokens"]

        text = "".join(texts).strip()
        merged_language = merge_languages(langs)

        t_total = time.perf_counter() - t_transcribe_start
        end_to_end_rtf = (t_total / audio_seconds) if audio_seconds > 0 else 0.0
        metrics = self.collect_metrics(
            feature_sec=agg["feature_sec"],
            encoder_sec=agg["encoder_sec"],
            prefill_sec=agg["prefill_sec"],
            decode_sec=agg["decode_sec"],
            detok_sec=agg["detok_sec"],
            prompt_tokens=agg["prompt_tokens"],
            generated_tokens=agg["generated_tokens"],
            encoder_tokens=agg["encoder_tokens"],
        )
        metrics["audio_duration_sec"] = audio_seconds
        metrics["model_load_sec"] = self.t_model_load
        metrics["end_to_end_sec"] = t_total
        metrics["rtf"] = end_to_end_rtf
        if merged_language:
            metrics["language"] = merged_language

        yield metrics
        yield text

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        removed = await registry.register_unload(model_name)
        self.enc_model = None
        self.emb_model = None
        self.dec_model = None
        self.dec_request = None
        gc.collect()
        logger.info(f"[{self.load_config.model_name}] unloaded and memory cleaned up")
        return removed


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR OpenVINO inference")
    parser.add_argument("wav_path")
    parser.add_argument("--ov-dir", default="ov_model")
    parser.add_argument("--device", default="CPU")
    parser.add_argument(
        "--language",
        default=None,
        help="Optional forced language (e.g. English, Chinese)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens per chunk",
    )
    parser.add_argument(
        "--max-chunk-sec",
        type=float,
        default=30.0,
        help="Target audio chunk size in seconds for energy-based splitting",
    )
    parser.add_argument(
        "--search-expand-sec",
        type=float,
        default=5.0,
        help="Boundary search window (seconds) around each tentative chunk cut",
    )
    parser.add_argument(
        "--min-window-ms",
        type=float,
        default=100.0,
        help="Sliding energy window in ms for low-energy boundary detection",
    )
    args = parser.parse_args()

    load_config = ModelLoadConfig(
        model_path=args.ov_dir,
        model_name="qwen3-asr-cli",
        model_type=ModelType.QWEN3_ASR,
        engine=EngineType.OPENVINO,
        device=args.device,
        runtime_config={},
    )
    config = Qwen3ASRConfig(
        audio_base64=base64.b64encode(Path(args.wav_path).read_bytes()).decode("utf-8"),
        language=args.language,
        max_tokens=args.max_tokens,
        max_chunk_sec=args.max_chunk_sec,
        search_expand_sec=args.search_expand_sec,
        min_window_ms=args.min_window_ms,
    )
    model = OVQwen3ASR(load_config)
    model.load_model(load_config)

    async def _run() -> str:
        final_text = ""
        async for item in model.transcribe(config):
            if isinstance(item, str):
                final_text = item
        return final_text

    text = asyncio.run(_run())
    print(text)


if __name__ == "__main__":
    main()
