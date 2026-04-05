from __future__ import annotations

import asyncio
import base64
import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np
import openvino as ov
import soundfile as sf
from transformers import AutoTokenizer

from src.engine.openvino.qwen3_tts.qwen3_tts_helpers import (
    CODEC_BOS_ID,
    CODEC_EOS_ID,
    CODEC_NOTHINK_ID,
    CODEC_PAD_ID,
    CODEC_THINK_BOS_ID,
    CODEC_THINK_EOS_ID,
    CODEC_THINK_ID,
    CP_HEAD_DIM,
    CP_MAX_POS,
    CP_ROPE_THETA,
    ENC_INPUT_SR,
    HEAD_DIM,
    LANGUAGES,
    NUM_CODE_GROUPS,
    SPEAKERS,
    SPEECH_DECODER_SR,
    SUPPRESS_MASK,
    TALKER_MAX_POS,
    TALKER_ROPE_THETA,
    TTS_BOS_TOKEN_ID,
    TTS_EOS_TOKEN_ID,
    TTS_PAD_TOKEN_ID,
    H,
    Language,
    Speaker,
    _INSTRUCT_TMPL,
    _REF_TEXT_TMPL,
    _SYNTH_TMPL,
)
from src.server.model_registry import ModelRegistry
from src.server.models.openvino import OV_Qwen3TTSGenConfig
from src.server.models.registration import EngineType, ModelLoadConfig, ModelType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ICL speech decoder: last N reference frames as left context (matches upstream chunked_decode).
ICL_DECODER_LEFT_CONTEXT_FRAMES = 25


@dataclass
class TTSStreamChunk:
    """One decoded PCM segment from streaming synthesis (float32 mono, SPEECH_DECODER_SR Hz)."""

    audio: np.ndarray
    chunk_index: int
    is_final: bool


def _perf_add(perf: dict | None, key: str, dt: float) -> None:
    if perf is not None:
        perf[key] = perf.get(key, 0.0) + dt


class OVQwen3TTS:
    """Single engine serving all three Qwen3-TTS modes.

    The mode is determined by load_config.model_type:
      ModelType.QWEN3_TTS_CUSTOM_VOICE — predefined speaker + optional instruct
      ModelType.QWEN3_TTS_VOICE_DESIGN — free-form voice description
      ModelType.QWEN3_TTS_VOICE_CLONE  — reference audio + optional ICL transcript
    """

    def __init__(self, load_config: ModelLoadConfig):
        self.load_config = load_config
        self._text_model_c = None
        self._codec_emb_c = None
        self._cp_codec_emb_c = None
        self._decoder_c = None
        self._decoder_input_name = None
        self._talker_req = None
        self._cp_req = None
        self._speaker_enc_c = None
        self._speech_enc_c = None
        self.tokenizer = None
        self._mrope_cos = None
        self._mrope_sin = None
        self._cp_cos = None
        self._cp_sin = None
        self._loaded = False

    # ---- Lifecycle ----------------------------------------------------------

    def load_model(self, load_config: ModelLoadConfig) -> None:
        """Load and compile OV models.

        Core models (text_model, codec_embedding, talker, code_predictor,
        speech_decoder) are loaded for every model type. When *device* is GPU,
        talker (and text/codec stacks) use GPU; code_predictor, cp_codec_embedding,
        and speech_decoder use CPU. Voice-clone models (speaker_encoder, speech_encoder)
        follow *device*. model_type == ModelType.QWEN3_TTS_VOICE_CLONE loads encoders.
        """
        self.load_config = load_config
        p = Path(load_config.model_path)
        device = load_config.device
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(p / ".ov_cache")})

        self.tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)

        self._mrope_cos, self._mrope_sin = H.precompute_mrope(
            TALKER_MAX_POS, HEAD_DIM, TALKER_ROPE_THETA,
        )
        self._cp_cos, self._cp_sin = H.precompute_standard_rope(
            CP_MAX_POS, CP_HEAD_DIM, CP_ROPE_THETA,
        )

        self._text_model_c = core.compile_model(str(p / "text_model.xml"), device)
        self._codec_emb_c = core.compile_model(str(p / "codec_embedding.xml"), device)
        # Code predictor: many tiny inferences per frame; CPU avoids GPU launch/transfer overhead.
        self._cp_codec_emb_c = core.compile_model(str(p / "cp_codec_embedding.xml"), "CPU")
        # Speech decoder: single-shot vocoding; CPU fits typical sequence lengths without GPU overhead.
        self._decoder_c = core.compile_model(
            str(p / "speech_tokenizer" / "speech_decoder.xml"), "CPU",
        )
        self._decoder_input_name = self._decoder_c.input(0).get_any_name()

        talker_c = core.compile_model(str(p / "talker.xml"), device)
        self._talker_req = talker_c.create_infer_request()
        cp_c = core.compile_model(str(p / "code_predictor.xml"), "CPU")
        self._cp_req = cp_c.create_infer_request()
        if "GPU" in device:
            logger.info(
                f"[{load_config.model_name}] talker on {device}; "
                f"code_predictor, cp_codec_embedding, speech_decoder on CPU",
            )

        self._speaker_enc_c = None
        self._speech_enc_c = None
        if load_config.model_type == ModelType.QWEN3_TTS_VOICE_CLONE:
            self._speaker_enc_c = core.compile_model(
                str(p / "speaker_encoder.xml"), device,
            )
            self._speech_enc_c = core.compile_model(
                str(p / "speech_tokenizer" / "speech_encoder.xml"), device,
            )

        self._loaded = True
        logger.info(
            f"[{load_config.model_name}] loaded from {p}  device={device}  "
            f"model_type={load_config.model_type.value}"
        )

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        removed = await registry.register_unload(model_name)
        self._text_model_c = None
        self._codec_emb_c = None
        self._cp_codec_emb_c = None
        self._decoder_c = None
        self._decoder_input_name = None
        self._talker_req = None
        self._cp_req = None
        self._speaker_enc_c = None
        self._speech_enc_c = None
        self.tokenizer = None
        self._mrope_cos = None
        self._mrope_sin = None
        self._cp_cos = None
        self._cp_sin = None
        self._loaded = False
        gc.collect()
        logger.info(f"[{model_name}] unloaded and memory cleaned up")
        return removed

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ---- Public API ---------------------------------------------------------

    async def generate(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
        """Synthesise speech from *gen_config*. Returns (wav: float32, sample_rate: int)."""
        return await asyncio.to_thread(self._generate_sync, gen_config)

    def generate_stream(self, gen_config: OV_Qwen3TTSGenConfig) -> Iterator[TTSStreamChunk]:
        """Synchronous generator of float32 mono PCM chunks at SPEECH_DECODER_SR (drip-fed text)."""
        if not self._loaded:
            raise RuntimeError("Call load_model() before generate_stream()")
        gc = gen_config.model_copy(update={"non_streaming_mode": False})
        if self.load_config.model_type == ModelType.QWEN3_TTS_VOICE_CLONE:
            yield from self._generate_voice_clone_stream(gc)
        else:
            yield from self._generate_standard_stream(gc)

    def _generate_sync(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
        if not self._loaded:
            raise RuntimeError("Call load_model() before generate()")
        if self.load_config.model_type == ModelType.QWEN3_TTS_VOICE_CLONE:
            return self._generate_voice_clone(gen_config)
        return self._generate_standard(gen_config)

    # ---- Internal: standard generation (custom_voice / voice_design) --------

    def _generate_standard(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
        t_total = time.perf_counter()
        perf: dict = {}
        speaker = Speaker(gen_config.speaker) if gen_config.speaker else None
        language = Language(gen_config.language) if gen_config.language else None

        if self.load_config.model_type == ModelType.QWEN3_TTS_CUSTOM_VOICE:
            build_kw = dict(
                text=gen_config.input,
                speaker=speaker,
                language=language,
                instruct=gen_config.instruct,
            )
        else:  # VOICE_DESIGN
            build_kw = dict(
                text=gen_config.input,
                speaker=None,
                language=language,
                instruct=gen_config.voice_description,
            )

        t0 = time.perf_counter()
        inp = self._build_inputs(**build_kw, non_streaming_mode=gen_config.non_streaming_mode, perf=perf)
        logger.debug(f"[perf] build_inputs: {time.perf_counter() - t0:.3f}s")

        codes = self._run_loop(inp, gen_config, perf)

        if not codes:
            self._log_pipeline_summary(perf, t_total, wav_seconds=0.0, voice_clone=False)
            return np.zeros(0, dtype=np.float32), SPEECH_DECODER_SR

        wav = self._decode_codes(codes, perf)

        self._log_summary(codes, wav, t_total)
        self._log_pipeline_summary(perf, t_total, wav_seconds=float(len(wav) / SPEECH_DECODER_SR), voice_clone=False)
        return wav, SPEECH_DECODER_SR

    # ---- Internal: voice clone generation -----------------------------------

    def _generate_voice_clone(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
        t_total = time.perf_counter()
        perf: dict = {}
        language = Language(gen_config.language) if gen_config.language else None

        t0 = time.perf_counter()
        audio, audio_sr = H.decode_audio_b64(gen_config.ref_audio_b64)
        dt_dec = time.perf_counter() - t0
        _perf_add(perf, "audio_decode", dt_dec)
        logger.debug(f"[perf] audio_decode: {dt_dec:.3f}s")
        logger.debug(f"[info] ref audio: {len(audio) / audio_sr:.3f}s @ {audio_sr} Hz")

        t0 = time.perf_counter()
        speaker_embed = self._extract_speaker_embedding(audio, audio_sr, perf)
        logger.debug(f"[perf] speaker encoder: {time.perf_counter() - t0:.3f}s")

        use_icl = gen_config.ref_text is not None and not gen_config.x_vector_only
        ref_codes = None
        if use_icl:
            t0 = time.perf_counter()
            ref_codes = self._encode_audio(audio, audio_sr, perf)
            logger.debug(f"[perf] speech encoder (OV): {time.perf_counter() - t0:.3f}s")
            logger.debug(f"[info] ref_codes shape: {ref_codes.shape}")

        t0 = time.perf_counter()
        inp = self._build_inputs(
            text=gen_config.input,
            speaker_embed=speaker_embed,
            language=language,
            instruct=gen_config.instruct,
            non_streaming_mode=gen_config.non_streaming_mode,
            ref_text=gen_config.ref_text if use_icl else None,
            ref_codes=ref_codes,
            perf=perf,
        )
        logger.debug(f"[perf] build_inputs: {time.perf_counter() - t0:.3f}s")

        codes = self._run_loop(inp, gen_config, perf)

        if not codes:
            self._log_pipeline_summary(perf, t_total, wav_seconds=0.0, voice_clone=True)
            return np.zeros(0, dtype=np.float32), SPEECH_DECODER_SR

        if use_icl and ref_codes is not None:
            wav = self._decode_icl(codes, ref_codes, perf)
        else:
            wav = self._decode_codes(codes, perf)

        self._log_summary(codes, wav, t_total)
        self._log_pipeline_summary(perf, t_total, wav_seconds=float(len(wav) / SPEECH_DECODER_SR), voice_clone=True)
        return wav, SPEECH_DECODER_SR

    def _generate_standard_stream(self, gen_config: OV_Qwen3TTSGenConfig) -> Iterator[TTSStreamChunk]:
        t_total = time.perf_counter()
        perf: dict = {}
        speaker = Speaker(gen_config.speaker) if gen_config.speaker else None
        language = Language(gen_config.language) if gen_config.language else None

        if self.load_config.model_type == ModelType.QWEN3_TTS_CUSTOM_VOICE:
            build_kw = dict(
                text=gen_config.input,
                speaker=speaker,
                language=language,
                instruct=gen_config.instruct,
            )
        else:
            build_kw = dict(
                text=gen_config.input,
                speaker=None,
                language=language,
                instruct=gen_config.voice_description,
            )

        t0 = time.perf_counter()
        inp = self._build_inputs(**build_kw, non_streaming_mode=gen_config.non_streaming_mode, perf=perf)
        logger.debug(f"[perf] build_inputs: {time.perf_counter() - t0:.3f}s")

        n_samples = 0
        n_chunks = 0
        for chunk in self._run_loop_streaming(inp, gen_config, perf):
            n_samples += len(chunk.audio)
            n_chunks += 1
            yield chunk

        wav_sec = n_samples / SPEECH_DECODER_SR
        self._log_pipeline_summary(perf, t_total, wav_seconds=wav_sec, voice_clone=False)
        if n_samples > 0:
            logger.info(f"[info] streaming: {n_chunks} chunks -> {n_samples} samples ({wav_sec:.2f}s audio)")

    def _generate_voice_clone_stream(self, gen_config: OV_Qwen3TTSGenConfig) -> Iterator[TTSStreamChunk]:
        t_total = time.perf_counter()
        perf: dict = {}
        language = Language(gen_config.language) if gen_config.language else None

        t0 = time.perf_counter()
        audio, audio_sr = H.decode_audio_b64(gen_config.ref_audio_b64)
        dt_dec = time.perf_counter() - t0
        _perf_add(perf, "audio_decode", dt_dec)
        logger.debug(f"[perf] audio_decode: {dt_dec:.3f}s")

        t0 = time.perf_counter()
        speaker_embed = self._extract_speaker_embedding(audio, audio_sr, perf)
        logger.debug(f"[perf] speaker encoder: {time.perf_counter() - t0:.3f}s")

        use_icl = gen_config.ref_text is not None and not gen_config.x_vector_only
        ref_codes = None
        if use_icl:
            t0 = time.perf_counter()
            ref_codes = self._encode_audio(audio, audio_sr, perf)
            logger.debug(f"[perf] speech encoder (OV): {time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        inp = self._build_inputs(
            text=gen_config.input,
            speaker_embed=speaker_embed,
            language=language,
            instruct=gen_config.instruct,
            non_streaming_mode=gen_config.non_streaming_mode,
            ref_text=gen_config.ref_text if use_icl else None,
            ref_codes=ref_codes,
            perf=perf,
        )
        logger.debug(f"[perf] build_inputs: {time.perf_counter() - t0:.3f}s")

        n_samples = 0
        n_chunks = 0
        for chunk in self._run_loop_streaming(inp, gen_config, perf):
            n_samples += len(chunk.audio)
            n_chunks += 1
            yield chunk

        wav_sec = n_samples / SPEECH_DECODER_SR
        self._log_pipeline_summary(perf, t_total, wav_seconds=wav_sec, voice_clone=True)
        if n_samples > 0:
            logger.info(f"[info] streaming (voice_clone): {n_chunks} chunks -> {n_samples} samples ({wav_sec:.2f}s audio)")

    def _decode_icl(
        self,
        gen_codes: list[list[int]],
        ref_codes: np.ndarray,
        perf: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Decode with a short ref prefix (left context), then trim the context from output."""
        ref_2d = ref_codes[0]  # (T_ref, n_q)
        gen_2d = np.asarray(gen_codes, dtype=np.int64)
        context_size = min(ICL_DECODER_LEFT_CONTEXT_FRAMES, ref_2d.shape[0])
        context = ref_2d[-context_size:]
        combined = np.concatenate([context, gen_2d], axis=0)
        decoder_in = combined.T[np.newaxis]  # (1, n_q, T)
        _b, n_q, t_frames = decoder_in.shape
        if perf is not None:
            perf["_decoder_in_shape"] = (_b, n_q, t_frames)
        logger.debug(
            f"[info] speech_decoder icl: ref_frames={ref_2d.shape[0]} "
            f"context_frames={context_size} gen_frames={gen_2d.shape[0]} "
            f"combined={combined.shape[0]}",
        )
        logger.debug(f"[info] speech decoder input shape: ({_b}, {n_q}, {t_frames})")
        t0 = time.perf_counter()
        result = H.ov_call(self._decoder_c, {self._decoder_input_name: decoder_in})
        dt = time.perf_counter() - t0
        _perf_add(perf, "speech_decoder", dt)
        logger.debug(f"[perf] speech decoder (OV): {dt:.3f}s")
        full_wav = np.clip(result["waveform"].squeeze(), -1.0, 1.0).astype(np.float32)
        cut = int(context_size / combined.shape[0] * len(full_wav))
        return full_wav[cut:]

    # ---- OV model wrappers --------------------------------------------------

    def _text_model(self, ids: np.ndarray) -> np.ndarray:
        return H.ov_call(self._text_model_c, {"token_ids": ids})["projected"]

    def _codec_embed(self, ids: np.ndarray) -> np.ndarray:
        return H.ov_call(self._codec_emb_c, {"token_ids": ids})["embeddings"]

    def _cp_codec_embed(self, ids: np.ndarray, step_idx: int) -> np.ndarray:
        return H.ov_call(self._cp_codec_emb_c, {
            "token_ids": ids,
            "step_idx": np.array(step_idx, dtype=np.int64),
        })["embeddings"]

    def _talker_infer(self, embeds, cos, sin):
        r = H.ov_stateful_infer(self._talker_req, {
            "inputs_embeds": embeds, "cos": cos, "sin": sin,
            "beam_idx": np.array([0], dtype=np.int32),
        })
        return r["logits"], r["hidden"]

    def _cp_infer(self, embeds, cos, sin, gen_steps: int):
        r = H.ov_stateful_infer(self._cp_req, {
            "inputs_embeds": embeds, "cos": cos, "sin": sin,
            "generation_steps": np.array(gen_steps, dtype=np.int64),
            "beam_idx": np.array([0], dtype=np.int32),
        })
        return r["logits"], r["hidden"]

    def _decode_codes(self, codes: list[list[int]], perf: dict[str, float] | None = None) -> np.ndarray:
        arr = np.asarray(codes, dtype=np.int64)
        decoder_in = arr.T[np.newaxis]
        _b, n_q, t_frames = decoder_in.shape
        if perf is not None:
            perf["_decoder_in_shape"] = (_b, n_q, t_frames)
        logger.debug(f"[info] speech decoder input shape: ({_b}, {n_q}, {t_frames})")
        t0 = time.perf_counter()
        r = H.ov_call(self._decoder_c, {self._decoder_input_name: decoder_in})
        dt = time.perf_counter() - t0
        _perf_add(perf, "speech_decoder", dt)
        logger.debug(f"[perf] speech decoder (OV): {dt:.3f}s")
        return np.clip(r["waveform"].squeeze(), -1.0, 1.0).astype(np.float32)

    def _chunked_decode(
        self,
        chunk_codes: list[list[int]],
        prev_tail: list[list[int]] | None,
        left_ctx: int,
        perf: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Decode codec frames with optional left context from the previous chunk."""
        if not chunk_codes:
            return np.zeros(0, dtype=np.float32)
        if prev_tail is None or left_ctx <= 0:
            return self._decode_codes(chunk_codes, perf)
        prev_arr = np.asarray(prev_tail, dtype=np.int64)
        if prev_arr.size == 0:
            return self._decode_codes(chunk_codes, perf)
        cur = np.asarray(chunk_codes, dtype=np.int64)
        ctx_n = min(left_ctx, prev_arr.shape[0])
        context = prev_arr[-ctx_n:]
        combined = np.concatenate([context, cur], axis=0)
        decoder_in = combined.T[np.newaxis]
        _b, n_q, t_frames = decoder_in.shape
        if perf is not None:
            perf["_decoder_in_shape"] = (_b, n_q, t_frames)
        logger.debug(f"[info] speech decoder (chunked) shape: ({_b}, {n_q}, {t_frames})")
        t0 = time.perf_counter()
        r = H.ov_call(self._decoder_c, {self._decoder_input_name: decoder_in})
        dt = time.perf_counter() - t0
        _perf_add(perf, "speech_decoder", dt)
        logger.debug(f"[perf] speech decoder chunk (OV): {dt:.3f}s")
        full_wav = np.clip(r["waveform"].squeeze(), -1.0, 1.0).astype(np.float32)
        cut = int(ctx_n / combined.shape[0] * len(full_wav))
        return full_wav[cut:]

    # ---- Voice-clone specific OV calls --------------------------------------

    def _extract_speaker_embedding(
        self, audio: np.ndarray, sr: int, perf: dict[str, float] | None = None,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        mels = H.mel_spectrogram(audio, sr)  # (n_mels, T)
        t_mel = time.perf_counter() - t0
        _perf_add(perf, "speaker_mel", t_mel)
        logger.debug(f"[perf] speaker mel_spectrogram: {t_mel:.3f}s")
        mels_in = mels.T[np.newaxis].astype(np.float32)  # (1, T, n_mels)
        t0 = time.perf_counter()
        r = H.ov_call(self._speaker_enc_c, {"mels": mels_in})
        t_ov = time.perf_counter() - t0
        _perf_add(perf, "speaker_ov", t_ov)
        logger.debug(f"[perf] speaker encoder ov: {t_ov:.3f}s")
        return r["embedding"][:, np.newaxis, :]  # (1, 1, D)

    def _encode_audio(self, audio: np.ndarray, sr: int, perf: dict[str, float] | None = None) -> np.ndarray:
        audio = audio.astype(np.float32)
        t_rs = 0.0
        if sr != ENC_INPUT_SR:
            t0 = time.perf_counter()
            audio = librosa.resample(audio, orig_sr=sr, target_sr=ENC_INPUT_SR)
            t_rs = time.perf_counter() - t0
            _perf_add(perf, "speech_resample", t_rs)
            logger.debug(f"[perf] speech encoder resample: {t_rs:.3f}s ({sr} -> {ENC_INPUT_SR} Hz)")
        t0 = time.perf_counter()
        r = H.ov_call(self._speech_enc_c, {"audio": audio[np.newaxis]})
        t_ov = time.perf_counter() - t0
        _perf_add(perf, "speech_ov", t_ov)
        logger.debug(f"[perf] speech encoder ov: {t_ov:.3f}s")
        return r["codes"]  # (1, T_ref, n_q)

    # ---- Prefill assembly ---------------------------------------------------

    def _get_special_embeds(self, perf: dict[str, float] | None = None):
        ids = np.array([[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]], dtype=np.int64)
        t0 = time.perf_counter()
        e = self._text_model(ids)
        _perf_add(perf, "build_text_model", time.perf_counter() - t0)
        return e[:, 0:1, :], e[:, 1:2, :], e[:, 2:3, :]

    def _resolve_language_id(self, language: Language | None, speaker: Speaker | None) -> int | None:
        lang_id = LANGUAGES[language].codec_id if language is not None else None
        if language in (Language.CHINESE, None) and speaker is not None:
            dialect = SPEAKERS[speaker].dialect
            if dialect is not None:
                lang_id = LANGUAGES[dialect].codec_id
        return lang_id

    def _build_codec_control(
        self,
        language_id: int | None,
        speaker_embed: np.ndarray | None = None,
        speaker: Speaker | None = None,
        perf: dict[str, float] | None = None,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        if language_id is None:
            prefix_ids = np.array(
                [[CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID]], dtype=np.int64,
            )
        else:
            prefix_ids = np.array(
                [[CODEC_THINK_ID, CODEC_THINK_BOS_ID, language_id, CODEC_THINK_EOS_ID]],
                dtype=np.int64,
            )

        emb_prefix = self._codec_embed(prefix_ids)
        emb_suffix = self._codec_embed(
            np.array([[CODEC_PAD_ID, CODEC_BOS_ID]], dtype=np.int64),
        )

        spk = None
        if speaker_embed is not None:
            spk = speaker_embed
        elif speaker is not None:
            spk = self._codec_embed(
                np.array([[SPEAKERS[speaker].codec_id]], dtype=np.int64),
            )

        parts = [emb_prefix] + ([spk] if spk is not None else []) + [emb_suffix]
        out = np.concatenate(parts, axis=1)
        dt_cc = time.perf_counter() - t0
        _perf_add(perf, "build_codec_control", dt_cc)
        logger.debug(f"[perf] build_codec_control: {dt_cc:.3f}s")
        return out

    def _build_inputs(
        self,
        text: str,
        speaker: Speaker | None = None,
        speaker_embed: np.ndarray | None = None,
        language: Language | None = None,
        instruct: str | None = None,
        non_streaming_mode: bool = True,
        ref_text: str | None = None,
        ref_codes: np.ndarray | None = None,
        perf: dict[str, float] | None = None,
    ) -> dict:
        formatted = _SYNTH_TMPL.format(text=text)
        t0 = time.perf_counter()
        input_ids = self.tokenizer(formatted, return_tensors="np", padding=False)["input_ids"]
        _perf_add(perf, "build_tokenizer", time.perf_counter() - t0)

        tts_bos, tts_eos, tts_pad = self._get_special_embeds(perf)
        lang_id = self._resolve_language_id(language, speaker)
        codec_ctrl = self._build_codec_control(lang_id, speaker_embed, speaker, perf)

        def _tm(ids: np.ndarray) -> np.ndarray:
            t0 = time.perf_counter()
            r = self._text_model(ids)
            _perf_add(perf, "build_text_model", time.perf_counter() - t0)
            return r

        def _ce_misc(ids: np.ndarray) -> np.ndarray:
            t0 = time.perf_counter()
            r = self._codec_embed(ids)
            _perf_add(perf, "build_codec_embed_other", time.perf_counter() - t0)
            return r

        # Role prefix: <|im_start|>assistant\n (first 3 tokens)
        role = _tm(input_ids[:, :3])

        # Control signal: text-side padding + bos summed with codec-side embeddings
        n_codec = codec_ctrl.shape[1]
        text_side = np.concatenate(
            [np.tile(tts_pad, (1, n_codec - 2, 1)), tts_bos], axis=1,
        )
        control = text_side + codec_ctrl[:, :-1, :]
        talker = np.concatenate([role, control], axis=1)

        if instruct:
            t0 = time.perf_counter()
            inst_ids = self.tokenizer(
                _INSTRUCT_TMPL.format(instruct=instruct), return_tensors="np", padding=False,
            )["input_ids"]
            _perf_add(perf, "build_tokenizer", time.perf_counter() - t0)
            talker = np.concatenate([_tm(inst_ids), talker], axis=1)

        use_icl = ref_codes is not None and ref_text is not None

        if use_icl:
            t0 = time.perf_counter()
            ref_ids = self.tokenizer(
                _REF_TEXT_TMPL.format(ref_text=ref_text), return_tensors="np", padding=False,
            )["input_ids"]
            _perf_add(perf, "build_tokenizer", time.perf_counter() - t0)
            ref_text_ids = ref_ids[:, 3:-2]
            target_ids = input_ids[:, 3:-5]
            all_text_ids = np.concatenate([ref_text_ids, target_ids], axis=1)

            text_emb = _tm(all_text_ids)
            text_eos = np.concatenate([text_emb, tts_eos], axis=1)

            codec_bos_emb = _ce_misc(np.array([[CODEC_BOS_ID]], dtype=np.int64))
            ref_emb = self._embed_ref_codes(ref_codes[0], perf)
            codec_bos_ref = np.concatenate([codec_bos_emb, ref_emb], axis=1)

            text_block = text_eos + _ce_misc(
                np.full((1, text_eos.shape[1]), CODEC_PAD_ID, dtype=np.int64),
            )
            codec_block = codec_bos_ref + np.tile(tts_pad, (1, codec_bos_ref.shape[1], 1))

            final_bos = tts_pad + _ce_misc(
                np.array([[CODEC_BOS_ID]], dtype=np.int64),
            )
            talker = np.concatenate([talker, text_block, codec_block, final_bos], axis=1)
            trailing = tts_pad

        elif non_streaming_mode:
            text_ids = input_ids[:, 3:-5]
            text_emb = _tm(text_ids)
            text_eos = np.concatenate([text_emb, tts_eos], axis=1)
            codec_pad_seq = _ce_misc(
                np.full((1, text_eos.shape[1]), CODEC_PAD_ID, dtype=np.int64),
            )
            final_bos = tts_pad + _ce_misc(
                np.array([[CODEC_BOS_ID]], dtype=np.int64),
            )
            talker = np.concatenate([talker, text_eos + codec_pad_seq, final_bos], axis=1)
            trailing = tts_pad

        else:
            first = _tm(input_ids[:, 3:4])
            talker = np.concatenate([talker, first + codec_ctrl[:, -1:, :]], axis=1)
            remaining = _tm(input_ids[:, 4:-5])
            trailing = np.concatenate([remaining, tts_eos], axis=1)

        _b, seq_len, _h = talker.shape
        logger.debug(f"[info] inputs_embeds shape: batch={_b} seq_len={seq_len} hidden={_h}")
        if perf is not None:
            logger.debug(
                f"[perf] build_inputs components: tokenizer={perf.get('build_tokenizer', 0):.3f}s "
                f"text_model={perf.get('build_text_model', 0):.3f}s "
                f"codec_control={perf.get('build_codec_control', 0):.3f}s "
                f"codec_embed_other={perf.get('build_codec_embed_other', 0):.3f}s "
                f"ref_embed={perf.get('build_ref_embed', 0):.3f}s",
            )

        return {"inputs_embeds": talker, "trailing_text_hidden": trailing, "tts_pad_embed": tts_pad}

    def _embed_ref_codes(self, codes_2d: np.ndarray, perf: dict[str, float] | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        T = codes_2d.shape[0]
        result = self._codec_embed(codes_2d[:, 0].reshape(1, T).astype(np.int64))
        for i in range(1, codes_2d.shape[1]):
            result = result + self._cp_codec_embed(
                codes_2d[:, i].reshape(1, T).astype(np.int64), step_idx=i - 1,
            )
        dt = time.perf_counter() - t0
        _perf_add(perf, "build_ref_embed", dt)
        logger.debug(f"[perf] embed_ref_codes: {dt:.3f}s")
        logger.debug(f"[info] embed_ref_codes T_ref_frames={T}")
        return result

    # ---- Sub-code generation ------------------------------------------------

    def _generate_sub_codes(
        self,
        first_code_embed: np.ndarray,
        past_hidden: np.ndarray,
        gen_config: OV_Qwen3TTSGenConfig,
    ) -> tuple[list[int], np.ndarray, float, float]:
        num_sub = NUM_CODE_GROUPS - 1
        self._cp_req.reset_state()

        prefill = np.concatenate([past_hidden, first_code_embed], axis=1)
        cos, sin = H.slice_rope(self._cp_cos, self._cp_sin, 0, 2)
        t_pf0 = time.perf_counter()
        logits, _ = self._cp_infer(prefill, cos, sin, gen_steps=0)
        t_cp_prefill = time.perf_counter() - t_pf0

        t_dc0 = time.perf_counter()
        tid = H.sample_token(
            logits[0, -1, :],
            gen_config.subtalker_do_sample, gen_config.subtalker_top_k,
            gen_config.subtalker_top_p, gen_config.subtalker_temperature,
        )
        sub_codes = [tid]

        code_emb = self._cp_codec_embed(np.array([[tid]], dtype=np.int64), step_idx=0)
        embeds_sum = first_code_embed + code_emb
        cache_pos = 2

        for step in range(1, num_sub):
            cos, sin = H.slice_rope(self._cp_cos, self._cp_sin, cache_pos, 1)
            logits, _ = self._cp_infer(code_emb, cos, sin, gen_steps=step)

            tid = H.sample_token(
                logits[0, -1, :],
                gen_config.subtalker_do_sample, gen_config.subtalker_top_k,
                gen_config.subtalker_top_p, gen_config.subtalker_temperature,
            )
            sub_codes.append(tid)

            code_emb = self._cp_codec_embed(np.array([[tid]], dtype=np.int64), step_idx=step)
            embeds_sum = embeds_sum + code_emb
            cache_pos += 1

        t_cp_decode = time.perf_counter() - t_dc0
        return sub_codes, embeds_sum, t_cp_prefill, t_cp_decode

    # ---- Core generation loop -----------------------------------------------

    def _run_loop(
        self,
        inp: dict,
        gen_config: OV_Qwen3TTSGenConfig,
        perf: dict[str, float] | None = None,
    ) -> list[list[int]]:
        """Run the autoregressive talker + code-predictor loop.

        Returns:
            List of codec frame codes (each frame is a list of NUM_CODE_GROUPS ints).
        """
        embeds = inp["inputs_embeds"]
        trailing = inp["trailing_text_hidden"]
        pad_emb = inp["tts_pad_embed"]

        self._talker_req.reset_state()
        S = embeds.shape[1]
        cos, sin = H.slice_rope(self._mrope_cos, self._mrope_sin, 0, S)

        t0 = time.perf_counter()
        logits, hidden = self._talker_infer(embeds, cos, sin)
        dt_tp = time.perf_counter() - t0
        _perf_add(perf, "talker_prefill", dt_tp)
        if perf is not None:
            perf["_talker_prefill_S"] = S
        logger.debug(f"[perf] talker prefill (S={S}): {dt_tp:.3f}s")

        cache_pos = S
        first_logits = logits[0, -1, :].copy()
        first_logits[SUPPRESS_MASK] = -np.inf
        first_code = H.sample_token(
            first_logits, gen_config.do_sample, gen_config.top_k,
            gen_config.top_p, gen_config.temperature,
        )

        all_codes: list[list[int]] = []
        past_first: list[int] = []
        past_hidden = hidden[:, -1:, :]
        t_cp = t_talk = 0.0
        t_cp_pf = t_cp_dc = 0.0

        step = 0
        while step < gen_config.max_new_tokens:
            if first_code == CODEC_EOS_ID:
                break

            past_first.append(first_code)
            fc_emb = self._codec_embed(np.array([[first_code]], dtype=np.int64))

            t0 = time.perf_counter()
            subs, emb_sum, t_pf, t_dc = self._generate_sub_codes(fc_emb, past_hidden, gen_config)
            t_cp += time.perf_counter() - t0
            t_cp_pf += t_pf
            t_cp_dc += t_dc

            all_codes.append([first_code] + subs)

            next_emb = emb_sum
            if step < trailing.shape[1]:
                next_emb = next_emb + trailing[:, step : step + 1, :]
            else:
                next_emb = next_emb + pad_emb

            cos, sin = H.slice_rope(self._mrope_cos, self._mrope_sin, cache_pos, 1)
            t0 = time.perf_counter()
            logits, hidden = self._talker_infer(next_emb, cos, sin)
            t_talk += time.perf_counter() - t0

            cache_pos += 1
            step += 1

            sl = logits[0, -1, :].copy()
            sl[SUPPRESS_MASK] = -np.inf
            if gen_config.repetition_penalty != 1.0 and past_first:
                sl = H.apply_repetition_penalty(sl, past_first, gen_config.repetition_penalty)
            first_code = H.sample_token(
                sl, gen_config.do_sample, gen_config.top_k,
                gen_config.top_p, gen_config.temperature,
            )
            past_hidden = hidden[:, -1:, :]

        n = step
        if n > 0:
            dt = t_cp + t_talk
            pf = dt / n
            if perf is not None:
                perf["_num_frames"] = n
            _perf_add(perf, "decode_talker", t_talk)
            _perf_add(perf, "decode_cp_prefill", t_cp_pf)
            _perf_add(perf, "decode_cp_decode", t_cp_dc)
            _perf_add(perf, "decode_loop_total", dt)
            logger.debug(f"[perf] decode loop ({n} frames):")
            logger.debug(f"[perf]   code predictor:  total={t_cp:.3f}s  avg={t_cp/n:.3f}s")
            logger.debug(f"[perf]   talker decode:   total={t_talk:.3f}s  avg={t_talk/n:.3f}s")
            logger.debug(f"[perf]   cp_prefill:      total={t_cp_pf:.3f}s  avg={t_cp_pf/n:.3f}s")
            logger.debug(f"[perf]   cp_decode:       total={t_cp_dc:.3f}s  avg={t_cp_dc/n:.3f}s")
            logger.debug(f"[perf]   per frame:       {pf:.3f}s  ({1/pf:.1f} fps)")
            logger.debug(f"[perf]   throughput:      {n * NUM_CODE_GROUPS / dt:.1f} tokens/s")

        return all_codes

    def _run_loop_streaming(
        self,
        inp: dict,
        gen_config: OV_Qwen3TTSGenConfig,
        perf: dict[str, float] | None = None,
    ) -> Iterator[TTSStreamChunk]:
        """Autoregressive loop like `_run_loop`, yielding decoded PCM at chunk boundaries."""
        chunk_size = max(1, gen_config.stream_chunk_frames)
        left_ctx = max(0, gen_config.stream_left_context)

        embeds = inp["inputs_embeds"]
        trailing = inp["trailing_text_hidden"]
        pad_emb = inp["tts_pad_embed"]

        self._talker_req.reset_state()
        S = embeds.shape[1]
        cos, sin = H.slice_rope(self._mrope_cos, self._mrope_sin, 0, S)

        t0 = time.perf_counter()
        logits, hidden = self._talker_infer(embeds, cos, sin)
        dt_tp = time.perf_counter() - t0
        _perf_add(perf, "talker_prefill", dt_tp)
        if perf is not None:
            perf["_talker_prefill_S"] = S
        logger.debug(f"[perf] talker prefill (S={S}): {dt_tp:.3f}s")

        cache_pos = S
        first_logits = logits[0, -1, :].copy()
        first_logits[SUPPRESS_MASK] = -np.inf
        first_code = H.sample_token(
            first_logits, gen_config.do_sample, gen_config.top_k,
            gen_config.top_p, gen_config.temperature,
        )

        buffer: list[list[int]] = []
        prev_tail: list[list[int]] | None = None
        chunk_index = 0
        past_first: list[int] = []
        past_hidden = hidden[:, -1:, :]
        t_cp = t_talk = 0.0
        t_cp_pf = t_cp_dc = 0.0

        step = 0
        while step < gen_config.max_new_tokens:
            if first_code == CODEC_EOS_ID:
                break

            past_first.append(first_code)
            fc_emb = self._codec_embed(np.array([[first_code]], dtype=np.int64))

            t0 = time.perf_counter()
            subs, emb_sum, t_pf, t_dc = self._generate_sub_codes(fc_emb, past_hidden, gen_config)
            t_cp += time.perf_counter() - t0
            t_cp_pf += t_pf
            t_cp_dc += t_dc

            buffer.append([first_code] + subs)

            if len(buffer) >= chunk_size:
                to_decode = buffer[:chunk_size]
                pcm = self._chunked_decode(to_decode, prev_tail, left_ctx, perf)
                yield TTSStreamChunk(pcm, chunk_index, is_final=False)
                take = min(left_ctx, len(to_decode))
                prev_tail = to_decode[-take:] if take > 0 else None
                buffer = buffer[chunk_size:]
                chunk_index += 1

            next_emb = emb_sum
            if step < trailing.shape[1]:
                next_emb = next_emb + trailing[:, step : step + 1, :]
            else:
                next_emb = next_emb + pad_emb

            cos, sin = H.slice_rope(self._mrope_cos, self._mrope_sin, cache_pos, 1)
            t0 = time.perf_counter()
            logits, hidden = self._talker_infer(next_emb, cos, sin)
            t_talk += time.perf_counter() - t0

            cache_pos += 1
            step += 1

            sl = logits[0, -1, :].copy()
            sl[SUPPRESS_MASK] = -np.inf
            if gen_config.repetition_penalty != 1.0 and past_first:
                sl = H.apply_repetition_penalty(sl, past_first, gen_config.repetition_penalty)
            first_code = H.sample_token(
                sl, gen_config.do_sample, gen_config.top_k,
                gen_config.top_p, gen_config.temperature,
            )
            past_hidden = hidden[:, -1:, :]

        n = step
        if n > 0:
            dt = t_cp + t_talk
            pf = dt / n
            if perf is not None:
                perf["_num_frames"] = n
            _perf_add(perf, "decode_talker", t_talk)
            _perf_add(perf, "decode_cp_prefill", t_cp_pf)
            _perf_add(perf, "decode_cp_decode", t_cp_dc)
            _perf_add(perf, "decode_loop_total", dt)
            logger.debug(f"[perf] decode loop ({n} frames):")
            logger.debug(f"[perf]   code predictor:  total={t_cp:.3f}s  avg={t_cp/n:.3f}s")
            logger.debug(f"[perf]   talker decode:   total={t_talk:.3f}s  avg={t_talk/n:.3f}s")
            logger.debug(f"[perf]   cp_prefill:      total={t_cp_pf:.3f}s  avg={t_cp_pf/n:.3f}s")
            logger.debug(f"[perf]   cp_decode:       total={t_cp_dc:.3f}s  avg={t_cp_dc/n:.3f}s")
            logger.debug(f"[perf]   per frame:       {pf:.3f}s  ({1/pf:.1f} fps)")
            logger.debug(f"[perf]   throughput:      {n * NUM_CODE_GROUPS / dt:.1f} tokens/s")

        if buffer:
            pcm = self._chunked_decode(buffer, prev_tail, left_ctx, perf)
            yield TTSStreamChunk(pcm, chunk_index, is_final=True)

    # ---- Logging ------------------------------------------------------------

    def _log_pipeline_summary(
        self,
        perf: dict,
        t_total_start: float,
        wav_seconds: float,
        voice_clone: bool,
    ) -> None:
        wall = time.perf_counter() - t_total_start
        bt = float(perf.get("build_tokenizer", 0.0))
        btm = float(perf.get("build_text_model", 0.0))
        bcc = float(perf.get("build_codec_control", 0.0))
        bce = float(perf.get("build_codec_embed_other", 0.0))
        bre = float(perf.get("build_ref_embed", 0.0))
        bi = bt + btm + bcc + bce + bre

        logger.info("[perf] === PIPELINE SUMMARY ===")
        if voice_clone:
            ad = float(perf.get("audio_decode", 0.0))
            if ad > 0:
                logger.info(f"[perf]   audio_decode:      {ad:.3f}s")
            sm = float(perf.get("speaker_mel", 0.0))
            so = float(perf.get("speaker_ov", 0.0))
            if sm > 0 or so > 0:
                logger.info(f"[perf]   speaker_encoder:   {sm + so:.3f}s  (mel: {sm:.3f}s, ov: {so:.3f}s)")
            sr_t = float(perf.get("speech_resample", 0.0))
            sp_o = float(perf.get("speech_ov", 0.0))
            if sp_o > 0 or sr_t > 0:
                logger.info(f"[perf]   speech_encoder:    {sr_t + sp_o:.3f}s  (resample: {sr_t:.3f}s, ov: {sp_o:.3f}s)")

        logger.info(
            f"[perf]   build_inputs:      {bi:.3f}s  "
            f"(tokenizer: {bt:.3f}s, text_model: {btm:.3f}s, codec_control: {bcc:.3f}s, "
            f"codec_embed: {bce:.3f}s, ref_embed: {bre:.3f}s)",
        )

        tp = float(perf.get("talker_prefill", 0.0))
        s_pf = perf.get("_talker_prefill_S")
        if tp > 0:
            if isinstance(s_pf, int):
                logger.info(f"[perf]   talker_prefill:    {tp:.3f}s  (S={s_pf})")
            else:
                logger.info(f"[perf]   talker_prefill:    {tp:.3f}s")

        n_fr = int(perf.get("_num_frames", 0) or 0)
        dtot = float(perf.get("decode_loop_total", 0.0))
        dtalk = float(perf.get("decode_talker", 0.0))
        dcpf = float(perf.get("decode_cp_prefill", 0.0))
        dcpd = float(perf.get("decode_cp_decode", 0.0))
        if n_fr > 0 and dtot > 0:
            logger.info(f"[perf]   decode_loop:       {dtot:.3f}s  (N={n_fr} frames)")
            logger.info(
                f"[perf]     talker_decode:   {dtalk:.3f}s  (avg {1000 * dtalk / n_fr:.2f}ms/frame)",
            )
            logger.info(
                f"[perf]     cp_prefill:      {dcpf:.3f}s  (avg {1000 * dcpf / n_fr:.2f}ms/frame)",
            )
            logger.info(
                f"[perf]     cp_decode:       {dcpd:.3f}s  (avg {1000 * dcpd / n_fr:.2f}ms/frame)",
            )

        sd = float(perf.get("speech_decoder", 0.0))
        dshape = perf.get("_decoder_in_shape")
        if sd > 0:
            if isinstance(dshape, tuple) and len(dshape) == 3:
                b, nq, tt = dshape
                logger.info(
                    f"[perf]   speech_decoder:    {sd:.3f}s  (input shape: {b}x{nq}x{tt})",
                )
            else:
                logger.info(f"[perf]   speech_decoder:    {sd:.3f}s")

        rt = wav_seconds / wall if wall > 0 else 0.0
        logger.info(f"[perf]   TOTAL:             {wall:.3f}s  -> {wav_seconds:.2f}s audio ({rt:.2f}x realtime)")

    @staticmethod
    def _log_summary(codes: list, wav: np.ndarray, t_total_start: float):
        sr = SPEECH_DECODER_SR
        logger.info(f"[perf] total: {time.perf_counter() - t_total_start:.3f}s")
        logger.info(f"[info] {len(codes)} frames -> {len(wav)} samples ({len(wav)/sr:.2f}s audio)")


if __name__ == "__main__":
    # Voice-clone smoke test without the server: edit the paths/strings below, then run
    #   uv run python -m src.engine.openvino.qwen3_tts.qwen3_tts
    _ov_dir = Path(
        "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen3-TTS-OpenVINO/"
        "Qwen3-TTS-12Hz-Base-1.7B-INT8-OpenVINO",
    )
    _ref_wav = Path("/home/echo/Projects/OpenArc/elmo_sample.wav")
    _ref_text = (
        "Color? Red! [laughs] Or, or who's your best friend? Um, Elmo's pet goldfish, "
        "Dorothy. Is it like... what is it like living on Sesame Street? That's a good "
        "question. Awesome, baby! [laughs] Wait... Elmo's not supposed to be answering "
        "these yet. [laughs] Sorry! [laughs] Well... now, you can ask Elmo any question "
        "you want right here on YouTube using this..."
    )
    _synth_text = "Hello! This is a quick voice clone test without running the server. Add some more text to see if the voice clone is working correctly. And again, and again and again."
    _output_wav = Path("voice_clone_out.wav")
    _device = "GPU.2"

    if not _ref_wav.is_file():
        raise SystemExit(f"Reference audio not found: {_ref_wav}")
    if not _ov_dir.is_dir():
        raise SystemExit(f"OpenVINO model directory not found: {_ov_dir}")

    _ref_b64 = base64.b64encode(_ref_wav.read_bytes()).decode("utf-8")
    _load = ModelLoadConfig(
        model_path=str(_ov_dir),
        model_name="qwen3-tts-voice-clone-entry",
        model_type=ModelType.QWEN3_TTS_VOICE_CLONE,
        engine=EngineType.OPENVINO,
        device=_device,
        runtime_config={},
    )
    _gen = OV_Qwen3TTSGenConfig(
        input=_synth_text,
        ref_audio_b64=_ref_b64,
        ref_text=_ref_text,
        x_vector_only=False,
        language=None,
        instruct=None,
    )

    _engine = OVQwen3TTS(_load)
    _engine.load_model(_load)

    async def _entry_run() -> tuple[np.ndarray, int]:
        return await _engine.generate(_gen)

    _wav, _sr = asyncio.run(_entry_run())
    _output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(_output_wav), _wav, _sr, subtype="PCM_16")
    print(f"Wrote {_output_wav} ({len(_wav) / _sr:.2f}s @ {_sr} Hz)")
