# streaming_kokoro_async.py
"""
Streaming-only Kokoro + OpenVINO implementation.
Now uses asyncio.to_thread for non-blocking streaming inference.
"""

import asyncio
import contextlib
import gc
import hashlib
import json
import logging
import math
import re
import types

from pathlib import Path
from typing import AsyncIterator, NamedTuple

import openvino as ov
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from kokoro import istftnet
from kokoro.model import KModel


from src.server.model_registry import ModelRegistry
from src.server.schemas.registration import ModelLoadConfig
from src.server.schemas.modeling.contract_kokoro import OV_KokoroGenConfig

logger = logging.getLogger(__name__)

# On GPU the OpenVINO plugin compiles kernels for every new input shape (5-11 s
# each), and the vocoder's length changes with every sentence. So the vocoder is
# compiled once per frame bucket with a static shape and inputs are padded up to
# the next bucket. Static shapes also
# matter for speed: with dynamic shapes the plugin falls back to reference kernels
# for most of the graph. Each bucket holds its own weights and buffers, so the
# list stops at 512 frames (~12.8 s of audio, ~2.4 GB of VRAM in total); longer
# vocoder inputs are cut in a pause and decoded piecewise.
GPU_DECODER_BUCKETS = (64, 96, 128, 192, 256, 384, 512)
# Audio samples the decoder produces per frame (24 kHz).
SAMPLES_PER_FRAME = 600
# CPU threads for the OpenVINO front end on the GPU path.
FRONT_CPU_THREADS = 4
# When streaming, the first text chunk is cut to about this many characters
# (a short sentence) so the first audio is ready quickly; later chunks use the
# request's character_count_chunk.
STREAM_FIRST_CHUNK_CHARS = 120
# Bump when the converted graphs change, to invalidate cached IRs.
_IR_VERSION = 2


def _pad_reflect(x: torch.Tensor, length: int) -> torch.Tensor:
    """Pad the last axis to `length` by mirroring the signal.

    The decoder's AdaIN layers normalise over the whole time axis, so zero
    padding shifts their statistics and audibly changes short clips. Mirrored
    content keeps the statistics close to the unpadded input.
    """
    out = x
    while out.shape[-1] < length:
        out = torch.cat([out, out.flip(-1)], dim=-1)
    return out[..., :length]


def _f02sine_exact(self, f0_values):
    """SineGen._f02sine with explicit interpolate sizes.

    Upstream uses scale_factor=1/upsample_scale. Exported to OpenVINO, that
    float scale rounds the output length off by one above ~2^20 samples
    (~1536 frames, ~38 s), and the graph then fails with "Argument shapes are
    inconsistent". Integer sizes are exact at any length and identical below it.
    """
    if self.flag_for_pulse:
        return _ORIGINAL_F02SINE(self, f0_values)
    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    n = rad_values.shape[1]
    rad_values = F.interpolate(rad_values.transpose(1, 2), size=n // self.upsample_scale, mode="linear").transpose(1, 2)
    phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
    phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, size=n, mode="linear").transpose(1, 2)
    return torch.sin(phase)


_ORIGINAL_F02SINE = istftnet.SineGen._f02sine


@contextlib.contextmanager
def _exact_sinegen():
    istftnet.SineGen._f02sine = _f02sine_exact
    try:
        yield
    finally:
        istftnet.SineGen._f02sine = _ORIGINAL_F02SINE


class _PolyphaseConvTranspose1d(nn.Module):
    """ConvTranspose1d with stride s, rewritten as one Conv1d that produces the
    s output phases side by side, then interleaved. Mathematically identical.

    The GPU plugin runs ConvTranspose1d on a reference kernel; on Kokoro's two
    generator upsamplers that was half the vocoder's time. Conv1d runs on oneDNN.
    """

    def __init__(self, ct: nn.ConvTranspose1d):
        super().__init__()
        if ct.dilation[0] != 1:
            raise ValueError("dilated ConvTranspose1d is not supported")
        w = ct.weight.detach()  # (Cin, Cout/groups, K)
        cin, cout_g, kernel = w.shape
        stride, groups = ct.stride[0], ct.groups
        taps = math.ceil(kernel / stride)
        # phase r, tap j uses kernel index r + j * stride
        phases = torch.zeros(cin, cout_g, stride, taps)
        for r in range(stride):
            for j in range(taps):
                if r + j * stride < kernel:
                    phases[:, :, r, j] = w[:, :, r + j * stride]
        # Conv1d weight (groups * Cout/g * s, Cin/g, taps); taps flipped because
        # Conv1d is a correlation and the transposed conv is a true convolution.
        phases = phases.view(groups, cin // groups, cout_g, stride, taps).permute(0, 2, 3, 1, 4)
        self.weight = nn.Parameter(phases.reshape(groups * cout_g * stride, cin // groups, taps).flip(-1).contiguous(), requires_grad=False)
        self.bias = None
        if ct.bias is not None:
            self.bias = nn.Parameter(ct.bias.detach().repeat_interleave(stride), requires_grad=False)
        self.stride, self.padding, self.output_padding = stride, ct.padding[0], ct.output_padding[0]
        self.kernel, self.taps, self.groups, self.out_channels = kernel, taps, groups, groups * cout_g

    def forward(self, x):
        length = x.shape[-1]
        y = F.conv1d(F.pad(x, (self.taps - 1, self.taps - 1)), self.weight, self.bias, groups=self.groups)
        batch, _, steps = y.shape
        y = y.view(batch, self.out_channels, self.stride, steps).transpose(2, 3).reshape(batch, self.out_channels, steps * self.stride)
        out_len = (length - 1) * self.stride - 2 * self.padding + self.kernel + self.output_padding
        return y[..., self.padding : self.padding + out_len]


def _remove_weight_norm(module: nn.Module) -> None:
    for m in module.modules():
        for hook in list(m._forward_pre_hooks.values()):
            if type(hook).__name__ == "WeightNorm":
                nn.utils.remove_weight_norm(m)


def _resblock1_folded_forward(self, x, s):
    """AdaINResBlock1.forward after _fold_snake_alpha.

    Snake is x + (1/a) * sin(a x)^2 = (u + sin(u)^2) / a with u = a x. The a is
    folded into the AdaIN affine before it and the 1/a into the conv after it,
    which removes the per-channel multiplies the GPU ran on reference kernels.
    """
    for c1, c2, n1, n2 in zip(self.convs1, self.convs2, self.adain1, self.adain2):
        xt = n1(x, s)
        xt = xt + torch.sin(xt) ** 2
        xt = c1(xt)
        xt = n2(xt, s)
        xt = xt + torch.sin(xt) ** 2
        xt = c2(xt)
        x = xt + x
    return x


@torch.no_grad()
def _fold_snake_alpha(block: "istftnet.AdaINResBlock1") -> None:
    pairs = list(zip(block.convs1, block.adain1, block.alpha1)) + list(zip(block.convs2, block.adain2, block.alpha2))
    for conv, adain, alpha in pairs:
        a = alpha.detach().view(-1)
        channels = a.numel()
        weight, bias = adain.fc.weight, adain.fc.bias  # rows [0, C) gamma, [C, 2C) beta
        # a * ((1 + gamma) * n + beta) == (1 + gamma') * n + beta'
        weight[:channels] *= a[:, None]
        bias[:channels] = a * bias[:channels] + a - 1
        weight[channels:] *= a[:, None]
        bias[channels:] *= a
        conv.weight /= a.view(1, -1, 1)
    block.forward = types.MethodType(_resblock1_folded_forward, block)


def _prepare_decoder_for_gpu(decoder: nn.Module) -> None:
    """Rewrite the decoder in place into GPU-friendly, numerically equivalent ops."""
    _remove_weight_norm(decoder)
    generator = decoder.generator
    for i, up in enumerate(generator.ups):
        generator.ups[i] = _PolyphaseConvTranspose1d(up)
    for block in decoder.decode:
        if isinstance(getattr(block, "pool", None), nn.ConvTranspose1d):
            block.pool = _PolyphaseConvTranspose1d(block.pool)
    for block in list(generator.resblocks) + list(generator.noise_res):
        _fold_snake_alpha(block)


class _Front(nn.Module):
    """KModel.forward_with_tokens (kokoro 0.9.4) up to the decoder call, as a
    graph OpenVINO can run on CPU. Not registered on the model: it only exists
    for conversion."""

    def __init__(self, kmodel: KModel, style_dim: int):
        super().__init__()
        self.kmodel = kmodel
        self.style_dim = style_dim

    def forward(self, input_ids, ref_s, speed):
        km = self.kmodel
        tokens = input_ids.shape[-1]
        input_lengths = torch.full((1,), tokens, dtype=torch.long)
        text_mask = torch.gt(torch.arange(tokens).unsqueeze(0) + 1, input_lengths.unsqueeze(1))
        d_en = km.bert_encoder(km.bert(input_ids, attention_mask=(~text_mask).int())).transpose(-1, -2)
        s = ref_s[:, self.style_dim :]
        d = km.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = km.predictor.lstm(d)
        duration = torch.sigmoid(km.predictor.duration_proj(x)).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0)
        indices = torch.repeat_interleave(torch.arange(tokens), pred_dur)
        alignment = (indices.unsqueeze(0) == torch.arange(tokens).unsqueeze(1)).float().unsqueeze(0)
        en = d.transpose(-1, -2) @ alignment
        F0_pred, N_pred = km.predictor.F0Ntrain(en, s)
        asr = km.text_encoder(input_ids, input_lengths, text_mask) @ alignment
        return asr, F0_pred, N_pred, pred_dur


class StreamChunk(NamedTuple):
    audio: torch.Tensor
    chunk_text: str
    chunk_index: int
    total_chunks: int


class OV_Kokoro(KModel):
    """
    We subclass the KModel from Kokoro to use with OpenVINO inputs.

    CPU: the exported openvino_model.xml runs end to end.
    GPU: the text front end (BERT, duration and F0 predictors, text encoder)
    runs as an OpenVINO graph on CPU, and the vocoder runs on the GPU as one
    static-shape graph per frame bucket, rewritten so its ops land on oneDNN
    kernels. Nothing compiles per request.
    """

    def __init__(self, load_config: ModelLoadConfig):
        super().__init__()
        self.model = None
        self._device = None
        self._gpu_decoders = {}
        self._front_cpu = None
        self._buckets = GPU_DECODER_BUCKETS
        self._space_id = None
        self._clause_end_ids = set()
        self._pipelines = {}

    def load_model(self, load_config: ModelLoadConfig):
        self.model_path = Path(load_config.model_path)
        self._device = load_config.device

        with (self.model_path / "config.json").open("r", encoding="utf-8") as f:
            model_config = json.load(f)

        self.vocab = model_config["vocab"]
        self.context_length = model_config["plbert"]["max_position_embeddings"]

        core = ov.Core()
        if load_config.cache_dir:
            core.set_property({"CACHE_DIR": load_config.cache_dir})
        if load_config.runtime_config:
            core.set_property(load_config.runtime_config)

        if str(self._device).upper().startswith("GPU"):
            self._hidden_dim = model_config["hidden_dim"]
            self._style_dim = model_config["style_dim"]
            self._space_id = self.vocab.get(" ")
            self._clause_end_ids = {self.vocab[c] for c in ".!?;:," if c in self.vocab}
            front, decoder = self._gpu_irs(core, load_config.cache_dir, model_config)
            self._front_cpu = core.compile_model(front, "CPU", {"INFERENCE_NUM_THREADS": FRONT_CPU_THREADS})
            for frames in self._buckets:
                static = decoder.clone()
                static.reshape({
                    0: [1, self._hidden_dim, frames],
                    1: [1, 2 * frames],
                    2: [1, 2 * frames],
                    3: [1, self._style_dim],
                })
                self._gpu_decoders[frames] = core.compile_model(static, self._device)
            self._warm_buckets()
            self.model = self._gpu_decoders[self._buckets[-1]]
            # Build the English pipelines now; the first request in each
            # language would otherwise pay ~1 s for G2P and spaCy loading.
            for lang_code in ("a", "b"):
                self._pipeline(lang_code)
        else:
            self.model = core.compile_model(self.model_path / "openvino_model.xml", self._device)
        return self.model

    def _gpu_irs(self, core: ov.Core, cache_dir, model_config: dict) -> tuple[ov.Model, ov.Model]:
        """Convert the front end and the rewritten decoder, reusing cached IRs."""
        key = hashlib.sha256(f"{_IR_VERSION}:{json.dumps(model_config, sort_keys=True)}".encode()).hexdigest()[:16]
        cached = {name: Path(cache_dir) / f"kokoro_{name}_{key}.xml" for name in ("front", "decoder")} if cache_dir else {}
        if cached and all(p.exists() for p in cached.values()):
            return core.read_model(cached["front"]), core.read_model(cached["decoder"])

        frames, tokens = 100, 32
        with torch.no_grad():
            front = ov.convert_model(
                _Front(self, self._style_dim).eval(),
                example_input=(torch.ones(1, tokens, dtype=torch.long), torch.zeros(1, 2 * self._style_dim), torch.tensor(1.0)),
            )
        _prepare_decoder_for_gpu(self.decoder)
        example = (
            torch.zeros(1, self._hidden_dim, frames),
            torch.zeros(1, 2 * frames),
            torch.zeros(1, 2 * frames),
            torch.zeros(1, self._style_dim),
        )
        with _exact_sinegen(), torch.no_grad():
            decoder = ov.convert_model(self.decoder, example_input=example)
        for name, model in (("front", front), ("decoder", decoder)):
            if name in cached:
                cached[name].parent.mkdir(parents=True, exist_ok=True)
                ov.save_model(model, cached[name], compress_to_fp16=False)
        return front, decoder

    def _warm_buckets(self) -> None:
        """Run every bucket once so no request pays for first-inference setup."""
        for frames, decoder in self._gpu_decoders.items():
            decoder([
                torch.zeros(1, self._hidden_dim, frames).numpy(),
                torch.zeros(1, 2 * frames).numpy(),
                torch.zeros(1, 2 * frames).numpy(),
                torch.zeros(1, self._style_dim).numpy(),
            ])

    def _front(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float):
        asr, F0_pred, N_pred, pred_dur = self._front_cpu([input_ids.numpy(), ref_s.numpy(), torch.tensor(float(speed)).numpy()]).to_tuple()
        return (torch.from_numpy(asr), torch.from_numpy(F0_pred), torch.from_numpy(N_pred),
                ref_s[:, : self._style_dim], torch.from_numpy(pred_dur))

    def _decode_bucketed(self, asr, F0_pred, N_pred, style) -> torch.FloatTensor:
        frames = asr.shape[-1]
        bucket = next(b for b in self._buckets if b >= frames)
        audio = self._gpu_decoders[bucket]([
            _pad_reflect(asr, bucket).numpy(),
            _pad_reflect(F0_pred, 2 * bucket).numpy(),
            _pad_reflect(N_pred, 2 * bucket).numpy(),
            style.numpy(),
        ])[0]
        return torch.from_numpy(audio).reshape(-1)[: frames * SAMPLES_PER_FRAME]

    def _plan_cuts(self, input_ids: torch.LongTensor, pred_dur: torch.LongTensor) -> list[int]:
        """Frame indices at which to cut the vocoder input so every piece fits the
        largest bucket. Cuts go in the middle of a space token's frames, i.e. in a
        pause: preferably after sentence or clause punctuation, else at any word
        gap, else (no space in reach) at the bucket limit."""
        limit = self._buckets[-1]
        if int(pred_dur.sum()) <= limit:
            return []
        tokens = input_ids[0].tolist()
        ends = torch.cumsum(pred_dur, dim=0).tolist()
        starts = [0] + ends[:-1]
        total = ends[-1]
        clause, word = [], []
        for i in range(1, len(tokens) - 1):
            if tokens[i] == self._space_id:
                mid = (starts[i] + ends[i]) // 2
                (clause if tokens[i - 1] in self._clause_end_ids else word).append(mid)
        cuts, pos = [], 0
        while total - pos > limit:
            window = lambda c: pos < c <= pos + limit
            # a clause cut anywhere in the back 60% of the window beats a later word cut
            options = [c for c in clause if window(c) and c > pos + 0.4 * limit] or [c for c in word if window(c)]
            pos = max(options) if options else pos + limit
            cuts.append(pos)
        return cuts

    def _pipeline(self, lang_code: str):
        """One KPipeline per language. Building one loads the G2P and spaCy
        models, which costs ~0.9 s, so it must not happen per request."""
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            from kokoro.pipeline import KPipeline
            pipeline = KPipeline(model=self, lang_code=lang_code)
            self._pipelines[lang_code] = pipeline
        return pipeline
    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """Run the compiled OpenVINO model. Without this override KModel's
        PyTorch version runs on CPU and the compiled model is never called."""
        if self._gpu_decoders:
            asr, F0_pred, N_pred, style, pred_dur = self._front(input_ids, ref_s, speed)
            bounds = [0, *self._plan_cuts(input_ids, pred_dur), asr.shape[-1]]
            pieces = [
                self._decode_bucketed(asr[..., a:b], F0_pred[..., 2 * a : 2 * b], N_pred[..., 2 * a : 2 * b], style)
                for a, b in zip(bounds, bounds[1:])
            ]
            return torch.cat(pieces), pred_dur
        outputs = self.model([input_ids, ref_s, torch.tensor(speed)])
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])

    def _pipeline(self, lang_code: str):
        """One KPipeline per language. Building one loads the G2P and spaCy
        models, which costs ~0.9 s, so it must not happen per request."""
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            from kokoro.pipeline import KPipeline
            pipeline = KPipeline(model=self, lang_code=lang_code)
            self._pipelines[lang_code] = pipeline
        return pipeline

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister model from registry and free memory resources.

        Args:
            registry: ModelRegistry to unregister from
            model_name: Model identifier to unload

        Returns:
            True if the model was found and unregistered, else False.
        """
        # Clean up model resources
        if self.model is not None:
            del self.model
            self.model = None
        self._gpu_decoders = {}
        self._front_cpu = None
        self._pipelines.clear()

        # Unregister from registry
        removed = await registry.register_unload(model_name)
        
        # Force garbage collection to free memory
        gc.collect()
        
        return removed


    # Sentence end: .!? (optionally followed by quotes/brackets), then whitespace.
    _SENTENCE_RE = re.compile(r'(?<=[.!?…])["\')\]]*\s+')
    # Clause punctuation worth pausing on when a mid-sentence split is needed.
    _CLAUSE_PUNCT = ",;:\u2014\u2013-"
    # Prefer clause splits at least this far into the chunk to avoid
    # degenerate tiny heads (e.g. a comma at position 4).
    _MIN_CLAUSE_FRACTION = 0.3

    @staticmethod
    def _split_head(text: str, limit: int) -> tuple[str, str]:
        """Split off a head of at most `limit` chars, preferring a clause
        boundary, then a word boundary, then a hard cut. Keeps punctuation
        on the head so the model still hears the pause."""
        head = text[:limit]
        cut = -1
        best = max(head.rfind(p) for p in OV_Kokoro._CLAUSE_PUNCT)
        if best >= int(limit * OV_Kokoro._MIN_CLAUSE_FRACTION):
            cut = best + 1
        else:
            cut = head.rfind(" ")
            if cut <= 0:
                cut = limit
        return text[:cut].strip(), text[cut:].strip()

    def make_chunks(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks of at most `chunk_size` characters.

        Boundary preference: paragraph (blank line) -> sentence -> clause
        (comma/semicolon/colon/dash) -> word -> hard cut. Guaranteed: every
        returned chunk respects the size limit and no character is ever
        dropped (concatenation reproduces the input modulo whitespace).
        """
        if not text or not text.strip():
            return []
        if len(text.strip()) <= chunk_size:
            return [text.strip()]

        # Paragraph breaks are hard boundaries; sentences within paragraphs.
        segments: list[str] = []
        for paragraph in re.split(r'\n\s*\n+|\n', text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            segments.extend(s for s in self._SENTENCE_RE.split(paragraph) if s.strip())

        chunks: list[str] = []
        current = ""

        for seg in segments:
            if len(seg) > chunk_size:
                # Oversized segment: flush buffer, then carve off heads until
                # the remainder fits. The tail becomes the new buffer so it
                # can merge with the next sentence.
                if current:
                    chunks.append(current)
                    current = ""
                while len(seg) > chunk_size:
                    head, seg = self._split_head(seg, chunk_size)
                    chunks.append(head)
            if not current:
                current = seg
            elif len(current) + 1 + len(seg) <= chunk_size:
                current = f"{current} {seg}"
            else:
                chunks.append(current)
                current = seg

        if current:
            chunks.append(current)

        return chunks

    def _stream_chunks(self, text: str, chunk_size: int) -> list[str]:
        """make_chunks, except the first chunk is cut down to a short sentence or
        clause so a streaming client gets audio as soon as possible."""
        chunks = self.make_chunks(text, chunk_size)
        if not chunks or len(chunks[0]) <= STREAM_FIRST_CHUNK_CHARS:
            return chunks
        head = self.make_chunks(chunks[0], STREAM_FIRST_CHUNK_CHARS)
        rest = " ".join(head[1:])
        return head[:1] + (self.make_chunks(rest, chunk_size) if rest else []) + chunks[1:]

    async def chunk_forward_pass(
        self, config: OV_KokoroGenConfig
    ) -> AsyncIterator[StreamChunk]:
        """
        Async generator yielding audio chunks from text.
        Uses asyncio.to_thread to offload inference calls.
        """
        pipeline = self._pipeline(config.lang_code.value)

        # Resolve the voice once. If voice_blend is set, this returns a
        # blended FloatTensor; otherwise the plain voice name.
        voice_arg = self._resolve_voice(config, pipeline)

        if getattr(config, "stream", False):
            text_chunks = self._stream_chunks(config.input, config.character_count_chunk)
        else:
            text_chunks = self.make_chunks(config.input, config.character_count_chunk)
        total_chunks = len(text_chunks)

        for idx, chunk_text in enumerate(text_chunks):

            def infer_on_chunk():
                """Blocking inference run in background thread."""
                with torch.no_grad():
                    infer = pipeline(chunk_text, voice=voice_arg, speed=config.speed)
                    if not hasattr(infer, "__iter__"):
                        return torch.as_tensor(infer.audio)
                    # KPipeline packs text into <=context_length phoneme
                    # buckets and yields one result per bucket. Consume ALL
                    # of them; taking only the first silently drops audio.
                    parts = [torch.as_tensor(r.audio) for r in infer]
                    if not parts:
                        return None
                    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

            # Run blocking inference off the main loop
            audio = await asyncio.to_thread(infer_on_chunk)
            if audio is None:
                continue

            yield StreamChunk(
                audio=audio,
                chunk_text=chunk_text,
                chunk_index=idx,
                total_chunks=total_chunks,
            )

    @staticmethod
    def _parse_blend(blend: str) -> list[tuple[str, float]]:
        """Parse a blend string into [(name, weight)] with weights normalised
        to sum to 1.0. Missing weights default to 1.0, so bare comma lists
        become equal-weight averages. Names are validated upstream."""
        items: list[tuple[str, float]] = []
        for part in blend.split(","):
            part = part.strip()
            if not part:
                continue
            name, _, weight = part.partition(":")
            name = name.strip()
            try:
                w = float(weight) if weight.strip() else 1.0
            except ValueError:
                w = 1.0
            items.append((name, max(0.0, w)))
        total = sum(w for _, w in items) or 1.0
        return [(n, w / total) for n, w in items]

    def _resolve_voice(self, config: "OV_KokoroGenConfig", pipeline):
        """Return the voice argument for KPipeline. Plain voice name when
        voice_blend is unset, otherwise a weighted-sum FloatTensor of the
        named voicepacks."""
        if not getattr(config, "voice_blend", None):
            return config.voice.value if hasattr(config.voice, "value") else config.voice
        components = self._parse_blend(config.voice_blend)
        if len(components) == 1:
            return components[0][0]
        packs = [pipeline.load_single_voice(n) * w for n, w in components]
        return torch.stack(packs).sum(dim=0)
