from __future__ import annotations

import asyncio
import gc
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import openvino as ov
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
from src.server.models.registration import ModelLoadConfig, ModelType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        speech_decoder) are loaded for every model type. Voice-clone models
        (speaker_encoder, speech_encoder) are loaded only when
        model_type == ModelType.QWEN3_TTS_VOICE_CLONE.
        """
        self.load_config = load_config
        p = Path(load_config.model_path)
        device = load_config.device
        core = ov.Core()

        self.tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)

        self._mrope_cos, self._mrope_sin = H.precompute_mrope(
            TALKER_MAX_POS, HEAD_DIM, TALKER_ROPE_THETA,
        )
        self._cp_cos, self._cp_sin = H.precompute_standard_rope(
            CP_MAX_POS, CP_HEAD_DIM, CP_ROPE_THETA,
        )

        self._text_model_c = core.compile_model(str(p / "text_model.xml"), device)
        self._codec_emb_c = core.compile_model(str(p / "codec_embedding.xml"), device)
        self._cp_codec_emb_c = core.compile_model(str(p / "cp_codec_embedding.xml"), device)
        self._decoder_c = core.compile_model(
            str(p / "speech_tokenizer" / "speech_decoder.xml"), device,
        )
        self._decoder_input_name = self._decoder_c.input(0).get_any_name()

        talker_c = core.compile_model(str(p / "talker.xml"), device)
        self._talker_req = talker_c.create_infer_request()
        cp_c = core.compile_model(str(p / "code_predictor.xml"), device)
        self._cp_req = cp_c.create_infer_request()

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

    def _generate_sync(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
        if not self._loaded:
            raise RuntimeError("Call load_model() before generate()")
        if self.load_config.model_type == ModelType.QWEN3_TTS_VOICE_CLONE:
            return self._generate_voice_clone(gen_config)
        return self._generate_standard(gen_config)

    # ---- Internal: standard generation (custom_voice / voice_design) --------

    def _generate_standard(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
        t_total = time.perf_counter()
        speaker = Speaker(gen_config.speaker) if gen_config.speaker else None
        language = Language(gen_config.language) if gen_config.language else None

        if self.load_config.model_type == ModelType.QWEN3_TTS_CUSTOM_VOICE:
            build_kw = dict(
                text=gen_config.text,
                speaker=speaker,
                language=language,
                instruct=gen_config.instruct,
            )
        else:  # VOICE_DESIGN
            build_kw = dict(
                text=gen_config.text,
                speaker=None,
                language=language,
                instruct=gen_config.voice_description,
            )

        t0 = time.perf_counter()
        inp = self._build_inputs(**build_kw, non_streaming_mode=gen_config.non_streaming_mode)
        logger.debug(f"[perf] build_inputs: {time.perf_counter() - t0:.3f}s")

        codes = self._run_loop(inp, gen_config)

        if not codes:
            return np.zeros(0, dtype=np.float32), SPEECH_DECODER_SR

        t0 = time.perf_counter()
        wav = self._decode_codes(codes)
        logger.debug(f"[perf] speech decoder (OV): {time.perf_counter() - t0:.3f}s")

        self._log_summary(codes, wav, t_total)
        return wav, SPEECH_DECODER_SR

    # ---- Internal: voice clone generation -----------------------------------

    def _generate_voice_clone(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
        t_total = time.perf_counter()
        language = Language(gen_config.language) if gen_config.language else None

        audio, audio_sr = H.decode_audio_b64(gen_config.ref_audio_b64)

        t0 = time.perf_counter()
        speaker_embed = self._extract_speaker_embedding(audio, audio_sr)
        logger.debug(f"[perf] speaker encoder: {time.perf_counter() - t0:.3f}s")

        use_icl = gen_config.ref_text is not None and not gen_config.x_vector_only
        ref_codes = None
        if use_icl:
            t0 = time.perf_counter()
            ref_codes = self._encode_audio(audio, audio_sr)
            logger.debug(f"[perf] speech encoder (OV): {time.perf_counter() - t0:.3f}s")
            logger.debug(f"[info] ref_codes shape: {ref_codes.shape}")

        t0 = time.perf_counter()
        inp = self._build_inputs(
            text=gen_config.text,
            speaker_embed=speaker_embed,
            language=language,
            instruct=gen_config.instruct,
            non_streaming_mode=gen_config.non_streaming_mode,
            ref_text=gen_config.ref_text if use_icl else None,
            ref_codes=ref_codes,
        )
        logger.debug(f"[perf] build_inputs: {time.perf_counter() - t0:.3f}s")

        codes = self._run_loop(inp, gen_config)

        if not codes:
            return np.zeros(0, dtype=np.float32), SPEECH_DECODER_SR

        t0 = time.perf_counter()
        if use_icl and ref_codes is not None:
            wav = self._decode_icl(codes, ref_codes)
        else:
            wav = self._decode_codes(codes)
        logger.debug(f"[perf] speech decoder (OV): {time.perf_counter() - t0:.3f}s")

        self._log_summary(codes, wav, t_total)
        return wav, SPEECH_DECODER_SR

    def _decode_icl(
        self,
        gen_codes: list[list[int]],
        ref_codes: np.ndarray,
    ) -> np.ndarray:
        """Decode with reference prefix, then trim the ref portion from output."""
        ref_2d = ref_codes[0]  # (T_ref, n_q)
        gen_2d = np.asarray(gen_codes, dtype=np.int64)
        combined = np.concatenate([ref_2d, gen_2d], axis=0)
        decoder_in = combined.T[np.newaxis]  # (1, n_q, T)
        result = H.ov_call(self._decoder_c, {self._decoder_input_name: decoder_in})
        full_wav = np.clip(result["waveform"].squeeze(), -1.0, 1.0).astype(np.float32)
        cut = int(ref_2d.shape[0] / combined.shape[0] * len(full_wav))
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

    def _decode_codes(self, codes: list[list[int]]) -> np.ndarray:
        arr = np.asarray(codes, dtype=np.int64)
        decoder_in = arr.T[np.newaxis]
        r = H.ov_call(self._decoder_c, {self._decoder_input_name: decoder_in})
        return np.clip(r["waveform"].squeeze(), -1.0, 1.0).astype(np.float32)

    # ---- Voice-clone specific OV calls --------------------------------------

    def _extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        mels = H.mel_spectrogram(audio, sr)  # (n_mels, T)
        mels_in = mels.T[np.newaxis].astype(np.float32)  # (1, T, n_mels)
        r = H.ov_call(self._speaker_enc_c, {"mels": mels_in})
        return r["embedding"][:, np.newaxis, :]  # (1, 1, D)

    def _encode_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = audio.astype(np.float32)
        if sr != ENC_INPUT_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=ENC_INPUT_SR)
        r = H.ov_call(self._speech_enc_c, {"audio": audio[np.newaxis]})
        return r["codes"]  # (1, T_ref, n_q)

    # ---- Prefill assembly ---------------------------------------------------

    def _get_special_embeds(self):
        ids = np.array([[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]], dtype=np.int64)
        e = self._text_model(ids)
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
    ) -> np.ndarray:
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
        return np.concatenate(parts, axis=1)

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
    ) -> dict:
        formatted = _SYNTH_TMPL.format(text=text)
        input_ids = self.tokenizer(formatted, return_tensors="np", padding=False)["input_ids"]

        tts_bos, tts_eos, tts_pad = self._get_special_embeds()
        lang_id = self._resolve_language_id(language, speaker)
        codec_ctrl = self._build_codec_control(lang_id, speaker_embed, speaker)

        # Role prefix: <|im_start|>assistant\n (first 3 tokens)
        role = self._text_model(input_ids[:, :3])

        # Control signal: text-side padding + bos summed with codec-side embeddings
        n_codec = codec_ctrl.shape[1]
        text_side = np.concatenate(
            [np.tile(tts_pad, (1, n_codec - 2, 1)), tts_bos], axis=1,
        )
        control = text_side + codec_ctrl[:, :-1, :]
        talker = np.concatenate([role, control], axis=1)

        if instruct:
            inst_ids = self.tokenizer(
                _INSTRUCT_TMPL.format(instruct=instruct), return_tensors="np", padding=False,
            )["input_ids"]
            talker = np.concatenate([self._text_model(inst_ids), talker], axis=1)

        use_icl = ref_codes is not None and ref_text is not None

        if use_icl:
            ref_ids = self.tokenizer(
                _REF_TEXT_TMPL.format(ref_text=ref_text), return_tensors="np", padding=False,
            )["input_ids"]
            ref_text_ids = ref_ids[:, 3:-2]
            target_ids = input_ids[:, 3:-5]
            all_text_ids = np.concatenate([ref_text_ids, target_ids], axis=1)

            text_emb = self._text_model(all_text_ids)
            text_eos = np.concatenate([text_emb, tts_eos], axis=1)

            codec_bos_emb = self._codec_embed(np.array([[CODEC_BOS_ID]], dtype=np.int64))
            ref_emb = self._embed_ref_codes(ref_codes[0])
            codec_bos_ref = np.concatenate([codec_bos_emb, ref_emb], axis=1)

            text_block = text_eos + self._codec_embed(
                np.full((1, text_eos.shape[1]), CODEC_PAD_ID, dtype=np.int64),
            )
            codec_block = codec_bos_ref + np.tile(tts_pad, (1, codec_bos_ref.shape[1], 1))

            final_bos = tts_pad + self._codec_embed(
                np.array([[CODEC_BOS_ID]], dtype=np.int64),
            )
            talker = np.concatenate([talker, text_block, codec_block, final_bos], axis=1)
            trailing = tts_pad

        elif non_streaming_mode:
            text_ids = input_ids[:, 3:-5]
            text_emb = self._text_model(text_ids)
            text_eos = np.concatenate([text_emb, tts_eos], axis=1)
            codec_pad_seq = self._codec_embed(
                np.full((1, text_eos.shape[1]), CODEC_PAD_ID, dtype=np.int64),
            )
            final_bos = tts_pad + self._codec_embed(
                np.array([[CODEC_BOS_ID]], dtype=np.int64),
            )
            talker = np.concatenate([talker, text_eos + codec_pad_seq, final_bos], axis=1)
            trailing = tts_pad

        else:
            first = self._text_model(input_ids[:, 3:4])
            talker = np.concatenate([talker, first + codec_ctrl[:, -1:, :]], axis=1)
            remaining = self._text_model(input_ids[:, 4:-5])
            trailing = np.concatenate([remaining, tts_eos], axis=1)

        return {"inputs_embeds": talker, "trailing_text_hidden": trailing, "tts_pad_embed": tts_pad}

    def _embed_ref_codes(self, codes_2d: np.ndarray) -> np.ndarray:
        T = codes_2d.shape[0]
        result = self._codec_embed(codes_2d[:, 0].reshape(1, T).astype(np.int64))
        for i in range(1, codes_2d.shape[1]):
            result = result + self._cp_codec_embed(
                codes_2d[:, i].reshape(1, T).astype(np.int64), step_idx=i - 1,
            )
        return result

    # ---- Sub-code generation ------------------------------------------------

    def _generate_sub_codes(
        self,
        first_code_embed: np.ndarray,
        past_hidden: np.ndarray,
        gen_config: OV_Qwen3TTSGenConfig,
    ) -> tuple[list[int], np.ndarray]:
        num_sub = NUM_CODE_GROUPS - 1
        self._cp_req.reset_state()

        prefill = np.concatenate([past_hidden, first_code_embed], axis=1)
        cos, sin = H.slice_rope(self._cp_cos, self._cp_sin, 0, 2)
        logits, _ = self._cp_infer(prefill, cos, sin, gen_steps=0)

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

        return sub_codes, embeds_sum

    # ---- Core generation loop -----------------------------------------------

    def _run_loop(
        self, inp: dict, gen_config: OV_Qwen3TTSGenConfig,
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
        logger.debug(f"[perf] talker prefill ({S}t): {time.perf_counter() - t0:.3f}s")

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

        step = 0
        while step < gen_config.max_new_tokens:
            if first_code == CODEC_EOS_ID:
                break

            past_first.append(first_code)
            fc_emb = self._codec_embed(np.array([[first_code]], dtype=np.int64))

            t0 = time.perf_counter()
            subs, emb_sum = self._generate_sub_codes(fc_emb, past_hidden, gen_config)
            t_cp += time.perf_counter() - t0

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
            logger.debug(f"[perf] decode loop ({n} frames):")
            logger.debug(f"[perf]   code predictor:  total={t_cp:.3f}s  avg={t_cp/n:.3f}s")
            logger.debug(f"[perf]   talker decode:   total={t_talk:.3f}s  avg={t_talk/n:.3f}s")
            logger.debug(f"[perf]   per frame:       {pf:.3f}s  ({1/pf:.1f} fps)")
            logger.debug(f"[perf]   throughput:      {n * NUM_CODE_GROUPS / dt:.1f} tokens/s")

        return all_codes

    # ---- Logging ------------------------------------------------------------

    @staticmethod
    def _log_summary(codes: list, wav: np.ndarray, t_total_start: float):
        sr = SPEECH_DECODER_SR
        logger.info(f"[perf] total: {time.perf_counter() - t_total_start:.3f}s")
        logger.info(f"[info] {len(codes)} frames -> {len(wav)} samples ({len(wav)/sr:.2f}s audio)")
