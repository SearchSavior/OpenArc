from __future__ import annotations

import base64
import io
import wave as wave_mod
from dataclasses import dataclass
from enum import StrEnum

import librosa
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants — fixed for the Qwen3-TTS OpenVINO checkpoint family
# ---------------------------------------------------------------------------

# Special token IDs
TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673
TTS_PAD_TOKEN_ID = 151671
CODEC_BOS_ID = 2149
CODEC_EOS_ID = 2150
CODEC_PAD_ID = 2148
CODEC_THINK_ID = 2154
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

# Talker architecture
NUM_CODE_GROUPS = 16
HIDDEN_SIZE = 2048
HEAD_DIM = 128
VOCAB_SIZE = 3072
TALKER_MAX_POS = 32768
TALKER_ROPE_THETA = 1_000_000.0
MROPE_SECTION = (24, 20, 20)

# Code predictor architecture
CP_HEAD_DIM = 128
CP_MAX_POS = 65536
CP_ROPE_THETA = 1_000_000.0

# Speech decoder
SPEECH_DECODER_SR = 24000

# Speaker encoder mel-spectrogram params
SE_SR = 24000
SE_N_FFT = 1024
SE_HOP = 256
SE_WIN = 1024
SE_N_MELS = 128
SE_FMIN = 0.0
SE_FMAX = 12000.0

# Speech encoder
ENC_INPUT_SR = 24000

# Prompt templates
_INSTRUCT_TMPL = "<|im_start|>user\n{instruct}<|im_end|>\n"
_SYNTH_TMPL = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
_REF_TEXT_TMPL = "<|im_start|>assistant\n{ref_text}<|im_end|>\n"

# Suppress mask: block last 1024 codec IDs except EOS
SUPPRESS_MASK = np.zeros(VOCAB_SIZE, dtype=bool)
for _i in range(VOCAB_SIZE - 1024, VOCAB_SIZE):
    if _i != CODEC_EOS_ID:
        SUPPRESS_MASK[_i] = True


# ---------------------------------------------------------------------------
# Language and Speaker enums + registries
# ---------------------------------------------------------------------------


class Language(StrEnum):
    CHINESE = "chinese"
    ENGLISH = "english"
    GERMAN = "german"
    ITALIAN = "italian"
    PORTUGUESE = "portuguese"
    SPANISH = "spanish"
    JAPANESE = "japanese"
    KOREAN = "korean"
    FRENCH = "french"
    RUSSIAN = "russian"
    BEIJING_DIALECT = "beijing_dialect"
    SICHUAN_DIALECT = "sichuan_dialect"


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    codec_id: int


LANGUAGES: dict[Language, LanguageInfo] = {
    Language.CHINESE:          LanguageInfo(codec_id=2055),
    Language.ENGLISH:          LanguageInfo(codec_id=2050),
    Language.GERMAN:           LanguageInfo(codec_id=2053),
    Language.ITALIAN:          LanguageInfo(codec_id=2070),
    Language.PORTUGUESE:       LanguageInfo(codec_id=2071),
    Language.SPANISH:          LanguageInfo(codec_id=2054),
    Language.JAPANESE:         LanguageInfo(codec_id=2058),
    Language.KOREAN:           LanguageInfo(codec_id=2064),
    Language.FRENCH:           LanguageInfo(codec_id=2061),
    Language.RUSSIAN:          LanguageInfo(codec_id=2069),
    Language.BEIJING_DIALECT:  LanguageInfo(codec_id=2074),
    Language.SICHUAN_DIALECT:  LanguageInfo(codec_id=2062),
}


class Speaker(StrEnum):
    SERENA = "serena"
    VIVIAN = "vivian"
    UNCLE_FU = "uncle_fu"
    RYAN = "ryan"
    AIDEN = "aiden"
    ONO_ANNA = "ono_anna"
    SOHEE = "sohee"
    ERIC = "eric"
    DYLAN = "dylan"


@dataclass(frozen=True, slots=True)
class SpeakerInfo:
    codec_id: int
    dialect: Language | None = None


SPEAKERS: dict[Speaker, SpeakerInfo] = {
    Speaker.SERENA:   SpeakerInfo(codec_id=3066),
    Speaker.VIVIAN:   SpeakerInfo(codec_id=3065),
    Speaker.UNCLE_FU: SpeakerInfo(codec_id=3010),
    Speaker.RYAN:     SpeakerInfo(codec_id=3061),
    Speaker.AIDEN:    SpeakerInfo(codec_id=2861),
    Speaker.ONO_ANNA: SpeakerInfo(codec_id=2873),
    Speaker.SOHEE:    SpeakerInfo(codec_id=2864),
    Speaker.ERIC:     SpeakerInfo(codec_id=2875, dialect=Language.SICHUAN_DIALECT),
    Speaker.DYLAN:    SpeakerInfo(codec_id=2878, dialect=Language.BEIJING_DIALECT),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OVQwen3TTSHelpers:
    """Static utility methods for sampling, RoPE, OV dispatch, and audio I/O."""

    # ---- Sampling -----------------------------------------------------------

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    @staticmethod
    def sample_token(
        logits: np.ndarray,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ) -> int:
        logits = logits.copy().astype(np.float32)
        if not do_sample:
            return int(np.argmax(logits))
        if temperature != 1.0:
            logits /= temperature
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = np.partition(logits, -top_k)[-top_k]
            logits[logits < threshold] = -np.inf
        if top_p < 1.0:
            idx = np.argsort(logits)[::-1]
            sl = logits[idx]
            probs = OVQwen3TTSHelpers.softmax(sl)
            cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
            sl[cutoff:] = -np.inf
            logits[idx] = sl
        probs = OVQwen3TTSHelpers.softmax(logits)
        return int(np.random.choice(len(probs), p=probs))

    @staticmethod
    def apply_repetition_penalty(
        logits: np.ndarray, past_tokens: list[int], penalty: float
    ) -> np.ndarray:
        for tid in set(past_tokens):
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty
        return logits

    # ---- RoPE ---------------------------------------------------------------

    @staticmethod
    def precompute_mrope(max_len: int, head_dim: int, theta: float = TALKER_ROPE_THETA):
        inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        pos = np.arange(max_len, dtype=np.float32)
        freqs = np.outer(pos, inv)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    @staticmethod
    def precompute_standard_rope(max_len: int, head_dim: int, theta: float = 10_000.0):
        inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        pos = np.arange(max_len, dtype=np.float32)
        freqs = np.outer(pos, inv)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    @staticmethod
    def slice_rope(cos, sin, start: int, length: int):
        c = cos[start : start + length][np.newaxis, np.newaxis]
        s = sin[start : start + length][np.newaxis, np.newaxis]
        return c, s

    # ---- OV dispatch --------------------------------------------------------

    @staticmethod
    def ov_call(compiled_model, inputs: dict) -> dict:
        result = compiled_model(inputs)
        return {out.get_any_name(): result[out] for out in compiled_model.outputs}

    @staticmethod
    def ov_stateful_infer(request, inputs: dict) -> dict:
        request.infer(inputs)
        return {
            out.get_any_name(): request.get_tensor(out.get_any_name()).data.copy()
            for out in request.model_outputs
        }

    # ---- Audio I/O ----------------------------------------------------------

    @staticmethod
    def load_audio_wav(path: str) -> tuple[np.ndarray, int]:
        with wave_mod.open(path, "r") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sw == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        else:
            raise ValueError(f"Unsupported sample width: {sw}")
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
        return samples, sr

    @staticmethod
    def decode_audio_b64(b64: str) -> tuple[np.ndarray, int]:
        data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr

    @staticmethod
    def mel_spectrogram(
        audio: np.ndarray,
        sr: int,
        target_sr: int = SE_SR,
        n_fft: int = SE_N_FFT,
        hop_size: int = SE_HOP,
        win_size: int = SE_WIN,
        n_mels: int = SE_N_MELS,
        fmin: float = SE_FMIN,
        fmax: float = SE_FMAX,
    ) -> np.ndarray:
        """Log-mel spectrogram -> (n_mels, T) float32."""
        audio = audio.astype(np.float32)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        pad = (n_fft - hop_size) // 2
        audio = np.pad(audio, (pad, pad), mode="reflect")
        stft = librosa.stft(
            audio, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
            window="hann", center=False,
        )
        mag = np.sqrt(stft.real ** 2 + stft.imag ** 2 + 1e-9).astype(np.float32)
        mel_basis = librosa.filters.mel(
            sr=target_sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
        ).astype(np.float32)
        return np.log(np.clip(mel_basis @ mag, 1e-5, None)).astype(np.float32)


H = OVQwen3TTSHelpers  # short alias used inside the engine
