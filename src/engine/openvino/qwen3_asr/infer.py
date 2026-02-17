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

import sys
import os
import json
import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov
import torch

from asr_utils import (
    MAX_ASR_INPUT_SECONDS,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)


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


@dataclass(frozen=True)
class Qwen3ASRConfig:
    wav_path: str
    ov_dir: str
    device: str
    language: Optional[str] = None
    max_tokens: int = 1024
    max_chunk_sec: float = 30.0
    search_expand_sec: float = 5.0
    min_window_ms: float = 100.0

    def validate(self) -> "Qwen3ASRConfig":
        if self.max_tokens <= 0:
            raise ValueError("--max-tokens must be positive")
        if self.max_chunk_sec <= 0:
            raise ValueError("--max-chunk-sec must be positive")
        if self.search_expand_sec <= 0:
            raise ValueError("--search-expand-sec must be positive")
        if self.min_window_ms <= 0:
            raise ValueError("--min-window-ms must be positive")
        return self


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

    def __init__(self, config: Qwen3ASRConfig):
        self.config = config.validate()
        self.ov_dir = Path(self.config.ov_dir)
        self.runtime_cfg = Qwen3ASRHelpers.hf_config(self.ov_dir / "config.json")
        self.language: Optional[str] = None
        if self.config.language:
            self.language = normalize_language_name(self.config.language)
            validate_language(self.language)

        self.chunk_size = self.runtime_cfg["enc_n_window"] * 2
        self.mel_filters = Qwen3ASRHelpers.compute_mel_filters()
        self.core = ov.Core()
        self.t_model_load = 0.0
        self.load_model()

    def load_model(self):
        print("Loading OpenVINO models...", file=sys.stderr)
        t_load_start = time.perf_counter()
        self.enc_model = self.core.compile_model(
            str(self.ov_dir / "audio_encoder.xml"),
            self.config.device,
        )
        self.emb_model = self.core.compile_model(
            str(self.ov_dir / "thinker_embeddings.xml"),
            self.config.device,
        )
        self.dec_model = self.core.compile_model(
            str(self.ov_dir / "decoder.xml"),
            self.config.device,
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

    async def audio_chunks(self, chunk_audio):
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
        self.dec_request.transcribe()
        logits = self.dec_request.get_output_tensor(0).data
        t_prefill = time.perf_counter() - t_prefill_start

        token = int(np.argmax(logits[0, 0]))
        generated = [token]

        max_tokens = self.config.max_tokens
        t_decode_start = time.perf_counter()
        for step in range(max_tokens - 1):
            if token in EOS_TOKEN_IDS:
                break
            pos = prompt_len + step
            embed = self._embed_tokens([token]).astype(np.float32)
            pos_id = np.array([[pos]], dtype=np.int64)

            self.dec_request.set_input_tensor(0, ov.Tensor(embed))
            self.dec_request.set_input_tensor(1, ov.Tensor(pos_id))
            self.dec_request.transcribe()
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

    async def transcribe(self) -> str:
        t_transcribe_start = time.perf_counter()
        audio_array = (await asyncio.to_thread(normalize_audios, self.config.wav_path))[0]

        audio_seconds = len(audio_array) / SAMPLE_RATE
        if audio_seconds <= 0:
            return ""

        max_chunk_sec = min(float(self.config.max_chunk_sec), float(MAX_ASR_INPUT_SECONDS))
        chunk_items = await asyncio.to_thread(
            split_audio_into_chunks,
            wav=audio_array,
            sr=SAMPLE_RATE,
            max_chunk_sec=max_chunk_sec,
            search_expand_sec=self.config.search_expand_sec,
            min_window_ms=self.config.min_window_ms,
        )
        print(f"Running {len(chunk_items)} chunk(s)...", file=sys.stderr)

        langs = []
        texts = []
        for idx, (chunk_wav, chunk_offset_sec) in enumerate(chunk_items):
            chunk_sec = len(chunk_wav) / SAMPLE_RATE
            print(
                f"Chunk {idx + 1}/{len(chunk_items)}: offset={chunk_offset_sec:.2f}s, duration={chunk_sec:.2f}s",
                file=sys.stderr,
            )
            raw, chunk_metrics = await self.audio_chunks(chunk_wav)
            print(
                "  feature={:.3f}s encoder={:.3f}s prefill={:.3f}s ({:.2f} tok/s) "
                "decode={:.3f}s ({:.2f} tok/s) detok={:.3f}s".format(
                    chunk_metrics["feature_sec"],
                    chunk_metrics["encoder_sec"],
                    chunk_metrics["prefill_sec"],
                    chunk_metrics["prefill_tok_s"],
                    chunk_metrics["decode_sec"],
                    chunk_metrics["decode_tok_s"],
                    chunk_metrics["detok_sec"],
                ),
                file=sys.stderr,
            )
            print(
                f"  prompt_tokens={chunk_metrics['prompt_tokens']} "
                f"generated_tokens={chunk_metrics['generated_tokens']} "
                f"encoder_tokens={chunk_metrics['encoder_tokens']}",
                file=sys.stderr,
            )
            lang, text = parse_asr_output(raw, language=self.language)
            langs.append(lang)
            if text:
                texts.append(text)

        text = "".join(texts).strip()
        merged_language = merge_languages(langs)

        t_total = time.perf_counter() - t_transcribe_start
        end_to_end_rtf = (t_total / audio_seconds) if audio_seconds > 0 else 0.0
        print("\n=== Performance Metrics ===", file=sys.stderr)
        print(f"Audio duration:        {audio_seconds:.3f} s", file=sys.stderr)
        print(f"Model load:            {self.t_model_load:.3f} s", file=sys.stderr)
        if merged_language:
            print(f"Detected language(s):  {merged_language}", file=sys.stderr)
        print(f"End-to-end (no load):  {t_total:.3f} s", file=sys.stderr)
        print(f"End-to-end RTF (no load): {end_to_end_rtf:.3f}x", file=sys.stderr)
        return text


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

    config = Qwen3ASRConfig(
        wav_path=args.wav_path,
        ov_dir=args.ov_dir,
        device=args.device,
        language=args.language,
        max_tokens=args.max_tokens,
        max_chunk_sec=args.max_chunk_sec,
        search_expand_sec=args.search_expand_sec,
        min_window_ms=args.min_window_ms,
    )
    text = asyncio.run(OVQwen3ASR(config).transcribe())
    print(text)


if __name__ == "__main__":
    main()
