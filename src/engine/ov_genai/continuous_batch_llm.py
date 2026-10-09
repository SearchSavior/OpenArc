"""Continuous-batching serving engine over openvino_genai.ContinuousBatchingPipeline.

Registered as engine ``ovgenai_cb`` (llm and vlm model types) as an opt-in
alternative to ``ovgenai``: requests overlap in OpenVINO's continuous
batching scheduler instead of serializing on the per-model queue worker.

Requires the ``nightly-ov`` dependency group (see docs/openvino-nightly.md):
the GenerationHandle API used here (add_request / step / can_read / read /
get_perf_metrics) only exists on openvino-genai nightlies, not on the pinned
stable release.
"""
import asyncio
import base64
import gc
import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino as ov
from openvino_genai import (
    ContinuousBatchingPipeline,
    GenerationConfig,
    GenerationStatus,
    SchedulerConfig,
    StreamerBase,
    StreamingStatus,
)
from PIL import Image
from transformers import AutoTokenizer, BatchEncoding

from src.engine.ov_genai.streamers import ensure_tool_call_parser, select_streamer
from src.engine.ov_genai.utils import (
    apply_temperature,
    check_vram_budget,
    extract_scheduler_config_from_loader,
    format_perf_metrics,
    load_draft_model,
)
from src.server.model_registry import ModelRegistry
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import ModelLoadConfig, ModelType
from src.server.utils.chat import flatten_message_content, flatten_messages
from src.server.utils.resolve_vlm_type import is_qwen3_5_architecture, resolve_vlm_vision_token

logger = logging.getLogger(__name__)


@dataclass
class _RequestState:
    """Per-request bookkeeping shared by the step loop and the consumer."""
    request_id: Optional[str]
    internal_id: int
    gen_config: OVGenAI_GenConfig
    streamer: StreamerBase
    done: "asyncio.Future[Tuple[Dict[str, Any], str]]"
    handle: Optional[Any] = None  # openvino_genai GenerationHandle
    generated_ids: List[int] = field(default_factory=list)
    cancel_requested: bool = False


def _retrieve_future_exception(fut: "asyncio.Future") -> None:
    # Retrieve terminal exceptions so requests abandoned by their consumer do
    # not log "Future exception was never retrieved"; awaiting consumers still
    # see the exception.
    if not fut.cancelled():
        fut.exception()


class OVGenAI_ContinuousBatch:
    def __init__(self, load_config: ModelLoadConfig):
        self.model = None
        self.encoder_tokenizer = None
        self.vision_token = None
        self.load_config = load_config
        self._default_chat_template_kwargs: dict = {}
        self.draft_model_loaded = False
        self.model_num_assistant_tokens = None
        self.model_assistant_confidence_threshold = None

        self._id_counter = itertools.count()
        self._requests: Dict[int, _RequestState] = {}
        self._by_request_id: Dict[str, _RequestState] = {}
        self._submit_queue: Optional["asyncio.Queue[_RequestState]"] = None
        self._wake_event: Optional[asyncio.Event] = None
        self._step_task: Optional[asyncio.Task] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._stopping = False

    @property
    def _is_vlm(self) -> bool:
        return self.load_config.model_type == ModelType.VLM

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        chat_template_kwargs: dict = {}
    ) -> ov.Tensor:
        """
        Convert a messages (list of {role, content}) into ov.Tensor using the cached AutoTokenizer
        and its chat template.
        """
        prompt_token_ids = self.encoder_tokenizer.apply_chat_template(
            flatten_messages(messages),
            tools=tools,
            add_generation_prompt=True,
            skip_special_tokens=True,
            return_tensors="np",
            **chat_template_kwargs,
            )
        if isinstance(prompt_token_ids, BatchEncoding):
            prompt_token_ids = prompt_token_ids['input_ids']
        return ov.Tensor(prompt_token_ids)

    def _vision_token_for_index(self, index: int) -> str:
        """
        Return the correctly formatted vision token for the given image index.
        Handles templates that may contain an index placeholder like '{i}'.
        """
        token_template = self.vision_token if self.vision_token is not None else ""
        if "{i}" in token_template:
            return token_template.replace("{i}", str(index))
        return token_template

    def prepare_vlm_inputs(self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        chat_template_kwargs: dict = {}
    ) -> Tuple[str, List[ov.Tensor]]:
        """
        Parse a messages list and prepare text prompt + image tensors for VLM inference.

        Returns:
            (tokenized_messages, ov_images)
        """
        images: List[Image.Image] = []
        text_messages: List[Dict[str, Any]] = []

        # Step 1: Extract text and images
        for message in messages:
            # Multimodal message (list of dict content items)
            if isinstance(message.get("content", ""), list):
                text_parts: List[str] = []

                for content_item in message["content"]:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "image_url"
                    ):
                        image_url = content_item.get("image_url", {})
                        # Check for embedded base64 data
                        if (
                            isinstance(image_url, dict)
                            and isinstance(image_url.get("url", ""), str)
                            and image_url["url"].startswith("data:image/")
                        ):
                            base64_data = image_url["url"].split(",", 1)
                            if len(base64_data) > 1:
                                image_data = base64.b64decode(base64_data[1])
                                image = Image.open(BytesIO(image_data)).convert("RGB")
                                images.append(image)

                                # Insert model-specific image token where this image appears
                                token_str = self._vision_token_for_index(len(images) - 1)
                                text_parts.append(f" {token_str} ")

                    # Handle text segments
                    elif isinstance(content_item, dict) and content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))

                # Combine extracted text back into a unified string
                text_message = message.copy()
                text_message["content"] = flatten_message_content(
                    " ".join([t for t in text_parts if isinstance(t, str)]) if text_parts else ""
                )
                text_messages.append(text_message)

            # Simple text-only message
            else:
                text_messages.append(
                    {**message, "content": flatten_message_content(message.get("content"))}
                )

        # Step 2: Build the chat template prompt using cached tokenizer
        text_messages = flatten_messages(text_messages)
        tokenized_messages: str = self.encoder_tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            tools=tools,
            add_generation_prompt=True,
            **{**self._default_chat_template_kwargs, **chat_template_kwargs},
        )

        # Step 3: Convert images to OpenVINO Tensors
        ov_images: List[ov.Tensor] = []
        for img in images:
            arr = np.array(img, dtype=np.uint8)
            tensor = ov.Tensor(arr)
            ov_images.append(tensor)

        return tokenized_messages, ov_images

    def _strip_stray_vision_tokens(self, prompt: str, ov_images: List[ov.Tensor]) -> str:
        """
        Enforce the vision-tag / image count invariant that OpenVINO's
        inputs_embedder checks (vision_sequence.size() == n_visions).

        See OVGenAI_VLM._strip_stray_vision_tokens: a text-only prompt that
        happens to mention the model's own vision token would otherwise abort
        with "The number of native vision tags must match the number of
        provided images/videos". Idempotent: a no-op when images exist or the
        token is absent.
        """
        if ov_images or not self.vision_token:
            return prompt
        token_str = self._vision_token_for_index(0)
        if not token_str or token_str not in prompt:
            return prompt
        stray_count = prompt.count(token_str)
        logger.warning(
            f"[{self.load_config.model_name}] prompt contains "
            f"{stray_count} native vision token(s) but no image was provided; "
            "stripping token(s) from the input, solves bug found in PR #169"
        )
        return prompt.replace(token_str, " ")

    def _resolve_prompt_and_images(
        self, gen_config: OVGenAI_GenConfig
    ) -> Tuple[str, List[ov.Tensor]]:
        """
        Build (prompt, images) for the pipeline: bench input_ids / raw prompt / chat messages.
        """
        if gen_config.input_ids:
            prompt = self.encoder_tokenizer.decode(gen_config.input_ids, skip_special_tokens=False)
            images: List[ov.Tensor] = []
        elif gen_config.prompt:
            prompt = gen_config.prompt
            images = []
        else:
            prompt, images = self.prepare_vlm_inputs(gen_config.messages, gen_config.tools, gen_config.chat_template_kwargs)
        return self._strip_stray_vision_tokens(prompt, images), images

    # ------------------------------------------------------------------
    # Generation consumer protocol (mirrors OVGenAI_LLM)
    # ------------------------------------------------------------------

    def generate_type(self, gen_config: OVGenAI_GenConfig):
        """
        Unified text generation method that routes to streaming or non-streaming
        based on the stream flag in gen_config. Both paths return an async iterator.

        Args:
            gen_config: Configuration containing the stream flag and other parameters

        Returns:
            - Non-streaming: async iterator yielding [metrics: dict, new_text: str]
            - Streaming: async iterator yielding token chunks (str)... then [metrics: dict]
        """
        if gen_config.stream:
            return self.generate_stream(gen_config)
        else:
            return self.generate_text(gen_config)

    async def generate_text(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """
        Async non-streaming text generation.
        Yields in order: metrics (dict), new_text (str).
        """
        state = await self._submit(gen_config)
        try:
            metrics, text = await state.done
        except Exception as e:
            logger.error(f"[{self.load_config.model_name}] Error during non-streaming generation: {e}", exc_info=True)
            raise
        logger.info(f"[{self.load_config.model_name}] Generation completed, generated {len(text)} characters")
        yield metrics
        yield text

    async def generate_stream(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Async streaming text generation.
        Yields token chunks (str or chat_delta dict) as they arrive, then metrics (dict).
        """
        state = await self._submit(gen_config)
        try:
            while True:
                chunk = await state.streamer.text_queue.get()
                if chunk is None:
                    break
                yield chunk
            # Stream fully drained: surface the terminal outcome. Awaiting the
            # future re-raises a per-request failure here instead of leaving
            # the HTTP client waiting on a stream that already ended.
            metrics, _ = await state.done
            yield metrics
        finally:
            if not state.done.done():
                # Consumer abandoned mid-stream (client disconnect,
                # cancellation): stop the backend generation so it does not
                # run to completion.
                self._cancel_state(state)

    async def cancel(self, request_id: str) -> bool:
        """
        Cancel an ongoing generation by request_id.

        Args:
            request_id: The request ID to cancel

        Returns:
            True if cancellation was triggered, False if request_id is unknown
        """
        state = self._by_request_id.get(request_id)
        if state is None:
            return False
        self._cancel_state(state)
        logger.info(f"[{self.load_config.model_name}] Cancellation triggered for request {request_id}")
        return True

    def collect_metrics(self, gen_config: OVGenAI_GenConfig, perf_metrics) -> Dict[str, Any]:
        """
        Collect and format performance metrics into a dictionary.
        """
        return format_perf_metrics(gen_config, perf_metrics)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    async def _submit(self, gen_config: OVGenAI_GenConfig) -> _RequestState:
        """Register a generation request and hand it to the step loop."""
        if self.model is None:
            raise RuntimeError(f"[{self.load_config.model_name}] model is not loaded")
        ensure_tool_call_parser(gen_config, self.load_config)
        self._ensure_step_loop()
        streamer = select_streamer(self.model.get_tokenizer(), gen_config)
        state = _RequestState(
            request_id=gen_config.request_id,
            internal_id=next(self._id_counter),
            gen_config=gen_config,
            streamer=streamer,
            done=asyncio.get_running_loop().create_future(),
        )
        state.done.add_done_callback(_retrieve_future_exception)
        self._requests[state.internal_id] = state
        if gen_config.request_id is not None:
            self._by_request_id[gen_config.request_id] = state
        await self._submit_queue.put(state)
        self._wake_event.set()
        return state

    def _ensure_step_loop(self) -> None:
        """Start the step loop lazily: load_model may run without a running loop."""
        if self._step_task is not None and not self._step_task.done():
            return
        self._stopping = False
        self._submit_queue = asyncio.Queue()
        self._wake_event = asyncio.Event()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"openarc-cb-{self.load_config.model_name}",
            )
        self._step_task = asyncio.create_task(self._step_loop())

    async def _step_loop(self) -> None:
        """Drive the ContinuousBatchingPipeline: admit submissions, step, drain handles."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                if self._stopping:
                    break

                # Admit newly submitted requests.
                while not self._submit_queue.empty():
                    state = self._submit_queue.get_nowait()
                    if state.cancel_requested:
                        self._fail_request(state, RuntimeError(
                            f"request {state.request_id} was cancelled before it could start"
                        ))
                        continue
                    try:
                        self._admit(state)
                    except Exception as e:
                        logger.error(
                            f"[{self.load_config.model_name}] failed to admit request {state.request_id}: {e}",
                            exc_info=True,
                        )
                        self._fail_request(state, e)

                active = [s for s in self._requests.values() if s.handle is not None]
                if not active:
                    # Park until the next submission wakes us. Re-check the
                    # queue after clearing so a submission racing the clear
                    # is not missed.
                    self._wake_event.clear()
                    if self._submit_queue.empty() and not self._stopping:
                        await self._wake_event.wait()
                    continue

                await loop.run_in_executor(self._executor, self.model.step)
                for state in active:
                    try:
                        self._drain_handle(state)
                    except Exception as e:
                        logger.error(
                            f"[{self.load_config.model_name}] failed to drain request {state.request_id}: {e}",
                            exc_info=True,
                        )
                        self._fail_request(state, e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # step() itself raising is model-fatal: fail every active request
            # (the worker layer unloads the model on hard inference errors).
            logger.error(
                f"[{self.load_config.model_name}] continuous batching step loop failed: {e}",
                exc_info=True,
            )
        finally:
            for state in list(self._requests.values()):
                self._fail_request(state, RuntimeError(
                    f"[{self.load_config.model_name}] generation stopped: "
                    "model is unloading or the batching loop failed"
                ))

    def _admit(self, state: _RequestState) -> None:
        """Tokenize and add one request to the pipeline."""
        gen_config = state.gen_config
        generation_config = self.create_generation_config(gen_config)

        if self._is_vlm:
            prompt, ov_images = self._resolve_prompt_and_images(gen_config)
            if ov_images:
                state.handle = self.model.add_request(
                    state.internal_id, prompt, ov_images, generation_config
                )
            else:
                state.handle = self.model.add_request(
                    state.internal_id, prompt, generation_config
                )
            return

        # LLM: pre-encoded input_ids, raw prompts, and chat messages
        if gen_config.input_ids:
            # Pre-encoded input IDs (used by /openarc/bench endpoint for benchmarking)
            prompt_token_ids = ov.Tensor(np.array(gen_config.input_ids, dtype=np.int64).reshape(1, -1))
            state.handle = self.model.add_request(state.internal_id, prompt_token_ids, generation_config)
        elif gen_config.prompt:
            # Raw text (used by /v1/completions endpoint); the pipeline tokenizes it
            state.handle = self.model.add_request(state.internal_id, gen_config.prompt, generation_config)
        else:
            # Chat template tokenization for messages (used by /v1/chat/completions endpoint)
            prompt_token_ids = self.prepare_inputs(gen_config.messages, gen_config.tools, gen_config.chat_template_kwargs)
            state.handle = self.model.add_request(state.internal_id, prompt_token_ids, generation_config)

    def _drain_handle(self, state: _RequestState) -> None:
        """Read available tokens from a handle and feed the request's streamer."""
        handle = state.handle
        # read() blocks until tokens are available, so only call it while
        # can_read() reports pending output.
        while handle.can_read():
            outputs = handle.read()
            for output in outputs.values():
                ids = list(output.generated_ids)
                if not ids:
                    continue
                state.generated_ids.extend(ids)
                status = state.streamer.write(ids)
                if status == StreamingStatus.CANCEL:
                    self._cancel_handle(state)
                    break
        gen_status = handle.get_status()
        if gen_status != GenerationStatus.RUNNING:
            self._finish_request(state, gen_status)

    def _finish_request(self, state: _RequestState, status: GenerationStatus) -> None:
        """Terminate a request whose handle left RUNNING."""
        try:
            state.streamer.end()
        except Exception as e:
            logger.warning(
                f"[{self.load_config.model_name}] streamer end failed for request {state.request_id}: {e}"
            )

        if status == GenerationStatus.IGNORED:
            # The scheduler dropped the request (per-request KV cache OOM).
            # Fail this request; the rest of the batch keeps serving.
            self._fail_request(state, RuntimeError(
                f"[{self.load_config.model_name}] request was ignored by the continuous "
                "batching scheduler (KV cache exhausted); retry with fewer concurrent "
                "requests or a smaller cache_size"
            ))
            return

        # Perf metrics are only available once the handle leaves RUNNING, and
        # may throw for CANCEL/IGNORED handles.
        try:
            perf_metrics = state.handle.get_perf_metrics()
            metrics = format_perf_metrics(state.gen_config, perf_metrics)
        except Exception as e:
            logger.warning(
                f"[{self.load_config.model_name}] perf metrics unavailable for request "
                f"{state.request_id} (status {status}): {e}"
            )
            metrics = {"stream": state.gen_config.stream}

        text = "" if state.gen_config.stream else self._assemble_text(state)
        if not state.done.done():
            state.done.set_result((metrics, text))
        self._drop_request(state)

    def _assemble_text(self, state: _RequestState) -> str:
        """Decode the full generated text for a non-streaming request."""
        # gemma4/museglimmer protocol tags are special=True: their tool
        # streamers keep the raw tagged text so the route-level
        # parse_generation can split reasoning/tool calls.
        raw_text = getattr(state.streamer, "raw_text", None)
        if raw_text is not None:
            return raw_text
        if not state.generated_ids:
            return ""
        decoder_tokenizer = self.model.get_tokenizer()
        return decoder_tokenizer.decode([state.generated_ids], skip_special_tokens=True)[0]

    def _fail_request(self, state: _RequestState, exc: BaseException) -> None:
        """Fail one request without disturbing the rest of the batch."""
        try:
            # Wake a streaming consumer blocked on the stream queue.
            state.streamer.text_queue.put_nowait(None)
        except Exception:
            pass
        if not state.done.done():
            state.done.set_exception(exc)
        if state.handle is not None:
            self._cancel_handle(state)
        self._drop_request(state)

    def _drop_request(self, state: _RequestState) -> None:
        self._requests.pop(state.internal_id, None)
        if state.request_id is not None:
            self._by_request_id.pop(state.request_id, None)

    def _cancel_state(self, state: _RequestState) -> None:
        state.cancel_requested = True
        state.streamer.cancel()
        if state.handle is not None:
            self._cancel_handle(state)

    def _cancel_handle(self, state: _RequestState) -> None:
        handle = state.handle
        if handle is None:
            return
        try:
            handle.cancel()
        except Exception as e:
            # Upstream bug openvinotoolkit/model_server#4428:
            # GenerationHandle.cancel() can SIGSEGV the whole process when it
            # races the scheduler under load. If cancel() raises, fall back to
            # stop(), which ends the request cleanly at the next step boundary
            # instead of tearing it down mid-step.
            logger.warning(
                f"[{self.load_config.model_name}] handle.cancel() failed for request "
                f"{state.request_id} ({e}); falling back to handle.stop()"
            )
            try:
                handle.stop()
            except Exception:
                logger.warning(
                    f"[{self.load_config.model_name}] handle.stop() also failed for "
                    f"request {state.request_id}",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self, loader: ModelLoadConfig):
        """Load the ContinuousBatchingPipeline and cache the tokenizer.

        Args:
            loader: ModelLoadConfig containing model_path, device, engine, and runtime_config.
        """
        logger.info(f"{loader.model_name} loading...")
        logger.info(f"{loader.model_type} on {loader.device} with {loader.runtime_config}")

        check_vram_budget(loader)

        # Load draft model for speculative decoding if provided
        (
            draft_model,
            self.draft_model_loaded,
            self.model_num_assistant_tokens,
            self.model_assistant_confidence_threshold,
        ) = load_draft_model(loader)

        scheduler_config = extract_scheduler_config_from_loader(loader)
        pipeline_kwargs = {**(loader.runtime_config or {})}
        if loader.cache_dir:
            pipeline_kwargs['CACHE_DIR'] = loader.cache_dir
        if draft_model is not None:
            pipeline_kwargs['draft_model'] = draft_model

        sched = scheduler_config.get("scheduler_config")
        if sched is None:
            sched = SchedulerConfig()

        self.model = ContinuousBatchingPipeline(
            loader.model_path,
            scheduler_config=sched,
            device=loader.device,
            properties=pipeline_kwargs,
        )

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(loader.model_path)

        if self._is_vlm:
            self.vision_token = resolve_vlm_vision_token(loader.model_path)
            # Auto-detect Qwen3.5 architecture and inject enable_thinking
            self._detect_chat_template_defaults(loader)

        logger.info(f"{loader.model_name} loaded successfully")

        # Warm up: one short generation so OpenVINO compiles its GPU kernels
        # at load time instead of on the first request. OPENARC_WARMUP=0 opts out.
        if os.getenv('OPENARC_WARMUP', '1') != '0':
            try:
                warm_cfg = GenerationConfig()
                warm_cfg.max_new_tokens = 8
                warm_cfg.do_sample = False
                warm_cfg.apply_chat_template = False
                if hasattr(self.model, "generate"):
                    self.model.generate("Hello", warm_cfg)
                else:
                    handle = self.model.add_request(-1, "Hello", warm_cfg)
                    while handle.get_status() == GenerationStatus.RUNNING:
                        self.model.step()
                logger.info(f"{loader.model_name} warm-up generation complete")
            except Exception as e:
                logger.warning(f"{loader.model_name} warm-up generation failed: {e}")

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister model from registry and free memory resources.

        Args:
            registry: ModelRegistry to unregister from
            model_name: Public model name to unload

        Returns:
            True if the model was found and unregistered, else False.
        """
        removed = await registry.register_unload(model_name)

        # Stop the batching loop: cancel every in-flight handle, then wake the
        # loop so it exits at the top of its next iteration (its finally
        # block fails any remaining requests, terminating their consumers).
        self._stopping = True
        for state in list(self._requests.values()):
            self._cancel_state(state)
        if self._wake_event is not None:
            self._wake_event.set()
        if self._step_task is not None:
            try:
                await self._step_task
            except asyncio.CancelledError:
                pass
            self._step_task = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._submit_queue = None
        self._wake_event = None

        if self.model is not None:
            del self.model
            self.model = None

        if self.encoder_tokenizer is not None:
            del self.encoder_tokenizer
            self.encoder_tokenizer = None

        if self.vision_token is not None:
            del self.vision_token
            self.vision_token = None

        gc.collect()
        logger.info(f"[{self.load_config.model_name}] unloaded successfully")
        return removed

    def _detect_chat_template_defaults(self, loader: ModelLoadConfig) -> None:
        """Read config.json and set default chat_template_kwargs for known architectures."""
        import json
        config_path = os.path.join(loader.model_path, "config.json")
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            if isinstance(architectures, list) and is_qwen3_5_architecture(architectures):
                self._default_chat_template_kwargs = {"enable_thinking": True}
                logger.info(f"{loader.model_name}: detected Qwen3.5 architecture, enabling thinking")
        except Exception as e:
            logger.debug(f"{loader.model_name}: could not detect architecture defaults: {e}")

    def create_generation_config(self, config: OVGenAI_GenConfig) -> GenerationConfig:
        """
        Converts the config received by the API to the OpenVino-compatible config.
        """
        generation_kwargs = self.model.get_config() if self.model else GenerationConfig()
        generation_kwargs.max_new_tokens = config.max_tokens
        apply_temperature(generation_kwargs, config.temperature)
        generation_kwargs.top_k = config.top_k
        generation_kwargs.top_p = config.top_p
        generation_kwargs.repetition_penalty = config.repetition_penalty
        # Prompts arrive fully templated (messages) or raw (prompt/input_ids);
        # the pipeline must not apply a chat template on top.
        generation_kwargs.apply_chat_template = False

        if config.seed:
            generation_kwargs.rng_seed = config.seed
        if config.frequency_penalty:
            generation_kwargs.frequency_penalty = config.frequency_penalty
        if config.presence_penalty:
            generation_kwargs.presence_penalty = config.presence_penalty

        # Add speculative decoding parameters (mutually exclusive per OpenVINO docs)
        if config.num_assistant_tokens is not None:
            generation_kwargs.num_assistant_tokens = config.num_assistant_tokens
        elif config.assistant_confidence_threshold is not None:
            generation_kwargs.assistant_confidence_threshold = config.assistant_confidence_threshold
        elif getattr(self, 'draft_model_loaded', False):
            if self.model_num_assistant_tokens is not None:
                generation_kwargs.num_assistant_tokens = self.model_num_assistant_tokens
            elif self.model_assistant_confidence_threshold is not None:
                generation_kwargs.assistant_confidence_threshold = self.model_assistant_confidence_threshold
            else:
                default_tokens = int(os.getenv('OPENARC_DEFAULT_NUM_ASSISTANT_TOKENS', '4'))
                generation_kwargs.num_assistant_tokens = default_tokens
        return generation_kwargs
