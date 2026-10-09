"""Strata proxy engine: front a remote Strata server's OpenAI-compatible API.

A "model" registered with engine=strata is not loaded locally; it is an HTTP
endpoint (https://github.com/Niko1221/Strata) serving /v1/chat/completions for
models OpenVINO cannot run (e.g. GGUF MoE models such as Qwen3.8-Flash-Next).

Configuration
-------------
``load_config.model_path`` carries the endpoint base URL (e.g.
``http://localhost:8080``) -- there is no on-disk IR, so the CLI skips the IR
file check for this engine and config loading never resolves the value as a
path. Alternatively ``runtime_config.endpoint`` overrides model_path when set
(useful when a placeholder model_path reads better in config.yaml).

``load_config.device`` is required by the schema but unused by this engine;
any placeholder string (e.g. "strata") is fine.

Registration
------------
``MODEL_CLASS_REGISTRY[(EngineType.STRATA, ModelType.LLM)]`` maps here. The
class subclasses OVGenAI_LLM purely so WorkerRegistry's isinstance routing
spawns an LLM queue worker without any worker changes; every load/generate
method is overridden and no OpenVINO pipeline is created.

Request mapping (OVGenAI_GenConfig -> OpenAI chat/completions body)
-------------------------------------------------------------------
messages pass through verbatim (the remote server applies the chat template);
a raw ``prompt`` becomes a single user message. temperature (0 = greedy
upstream), top_p, top_k, max_tokens, seed, frequency_penalty,
presence_penalty, repetition_penalty, tools and chat_template_kwargs are
forwarded. Streaming requests ask for ``stream_options.include_usage`` so the
final SSE event carries token counts.

Metrics
-------
OpenArc's collect_metrics shape, but measured client-side: TTFT is the wall
time from request send to the first content/reasoning chunk, throughputs are
derived from the upstream usage object and elapsed time. The 'proxy' key marks
the dict as client-measured rather than engine-reported.

Cancellation
------------
Per-request the active streaming httpx.Response is tracked by request_id;
cancel() closes it, which the remote server observes as a client disconnect
and aborts generation.
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from src.engine.ov_genai.llm import OVGenAI_LLM
from src.server.model_registry import ModelRegistry
from src.server.schemas.modeling.contract_ovgenai_llm_and_vlm import OVGenAI_GenConfig
from src.server.schemas.registration import ModelLoadConfig

logger = logging.getLogger(__name__)

# Long generations must not trip a read timeout; connect stays bounded so a
# dead endpoint fails fast instead of hanging a queue worker.
_CLIENT_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
_CLIENT_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)
_PROBE_TIMEOUT = 10.0


class StrataProxyLLM(OVGenAI_LLM):
    """Engine proxying generation to a remote Strata (OpenAI-compatible) server."""

    def __init__(self, load_config: ModelLoadConfig, transport: Optional[httpx.BaseTransport] = None):
        super().__init__(load_config)
        # transport: test hook (httpx.MockTransport works for sync and async clients)
        self._transport = transport
        self.endpoint: str = ""
        self.served_model_id: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._load_time_s: float = 0.0
        # request_id -> active streaming response (for cancel)
        self._active_streams: Dict[str, httpx.Response] = {}
        self._cancelled: set = set()

    # ------------------------------------------------------------------ load

    @staticmethod
    def _resolve_endpoint(loader: ModelLoadConfig) -> str:
        """Endpoint base URL from runtime_config.endpoint, else model_path."""
        endpoint = (loader.runtime_config or {}).get("endpoint") or loader.model_path
        if not isinstance(endpoint, str) or "://" not in endpoint:
            raise ValueError(
                f"[{loader.model_name}] engine=strata needs the Strata endpoint URL in "
                f"model_path (e.g. http://localhost:8080) or runtime_config.endpoint; "
                f"got {endpoint!r}"
            )
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/v1"):
            endpoint = endpoint[: -len("/v1")]
        return endpoint

    def load_model(self, loader: ModelLoadConfig) -> None:
        """Verify the endpoint is reachable and capture the served model id.

        Runs synchronously (the registry calls load_model via to_thread), so the
        reachability probe uses a short-lived sync client; the streaming client
        is only constructed here and first used inside the event loop.
        """
        self.endpoint = self._resolve_endpoint(loader)
        logger.info(f"{loader.model_name} probing Strata endpoint {self.endpoint} ...")

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=_PROBE_TIMEOUT, transport=self._transport) as probe:
                resp = probe.get(f"{self.endpoint}/v1/models")
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"[{loader.model_name}] Strata endpoint {self.endpoint} unreachable: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"[{loader.model_name}] Strata endpoint {self.endpoint} answered "
                f"GET /v1/models with status {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json().get("data") or []
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"[{loader.model_name}] Strata endpoint {self.endpoint} returned "
                f"non-JSON from /v1/models"
            ) from exc
        if not data:
            raise RuntimeError(
                f"[{loader.model_name}] Strata endpoint {self.endpoint} is reachable "
                f"but serves no models"
            )
        self.served_model_id = data[0].get("id")
        self._load_time_s = time.perf_counter() - start

        self._client = httpx.AsyncClient(
            timeout=_CLIENT_TIMEOUT,
            limits=_CLIENT_LIMITS,
            transport=self._transport,
        )
        logger.info(
            f"{loader.model_name} loaded successfully "
            f"(proxy to {self.endpoint}, serving model id '{self.served_model_id}')"
        )

    # -------------------------------------------------------------- generate

    def generate_type(self, gen_config: OVGenAI_GenConfig):
        """
        Route to streaming or non-streaming generation.

        Returns:
            - Non-streaming: async iterator yielding [metrics: dict, new_text: str]
            - Streaming: async iterator yielding str chunks / {"chat_delta": [...]}
              dicts, then a terminal metrics dict
        """
        if gen_config.stream:
            return self.generate_stream(gen_config)
        return self.generate_text(gen_config)

    def _build_request_body(self, gen_config: OVGenAI_GenConfig, stream: bool) -> Dict[str, Any]:
        if gen_config.input_ids:
            raise ValueError(
                "engine=strata does not accept pre-encoded input_ids "
                "(/openarc/bench is unsupported); send a prompt or messages"
            )
        if gen_config.messages:
            messages = gen_config.messages
        elif gen_config.prompt:
            messages = [{"role": "user", "content": gen_config.prompt}]
        else:
            raise ValueError("engine=strata request needs messages or a prompt")

        body: Dict[str, Any] = {
            "model": self.served_model_id,
            "messages": messages,
            "temperature": gen_config.temperature,  # 0 means greedy upstream
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "max_tokens": gen_config.max_tokens,
            "stream": stream,
        }
        if gen_config.seed is not None:
            body["seed"] = gen_config.seed
        if gen_config.repetition_penalty != 1.0:
            body["repetition_penalty"] = gen_config.repetition_penalty
        if gen_config.frequency_penalty is not None:
            body["frequency_penalty"] = gen_config.frequency_penalty
        if gen_config.presence_penalty is not None:
            body["presence_penalty"] = gen_config.presence_penalty
        if gen_config.tools:
            body["tools"] = gen_config.tools
        if gen_config.chat_template_kwargs:
            body["chat_template_kwargs"] = gen_config.chat_template_kwargs
        if stream:
            # Ask the server for a final usage event so token counts are real.
            body["stream_options"] = {"include_usage": True}
        return body

    async def _post_chat(self, body: Dict[str, Any]) -> httpx.Response:
        assert self._client is not None, "load_model must run before generate_type"
        url = f"{self.endpoint}/v1/chat/completions"
        try:
            resp = await self._client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"[{self.load_config.model_name}] Strata request failed: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise RuntimeError(
                f"[{self.load_config.model_name}] Strata answered status "
                f"{resp.status_code}: {resp.text[:500]}"
            )
        return resp

    async def generate_text(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[Dict[str, Any], str]]:
        """Non-streaming generation. Yields metrics (dict), then new_text (str)."""
        body = self._build_request_body(gen_config, stream=False)
        start = time.perf_counter()
        resp = await self._post_chat(body)
        elapsed = time.perf_counter() - start

        payload = resp.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""

        yield self.collect_metrics(
            gen_config,
            usage=payload.get("usage"),
            ttft_s=elapsed,
            elapsed_s=elapsed,
        )
        yield text

    async def generate_stream(self, gen_config: OVGenAI_GenConfig) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Streaming generation.

        Yields str content chunks as SSE deltas arrive, reasoning deltas as
        {"chat_delta": [{"reasoning_content": ...}]} (the item protocol
        InferWorker.infer_llm already forwards), then a terminal metrics dict.
        """
        body = self._build_request_body(gen_config, stream=True)
        request_id = gen_config.request_id
        assert self._client is not None, "load_model must run before generate_type"
        url = f"{self.endpoint}/v1/chat/completions"

        start = time.perf_counter()
        ttft_s: Optional[float] = None
        usage: Optional[Dict[str, Any]] = None

        try:
            async with self._client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    error_text = (await resp.aread()).decode(errors="replace")
                    raise RuntimeError(
                        f"[{self.load_config.model_name}] Strata answered status "
                        f"{resp.status_code}: {error_text[:500]}"
                    )
                if request_id:
                    self._active_streams[request_id] = resp

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug(f"[{self.load_config.model_name}] undecodable SSE line: {data[:200]}")
                        continue

                    if event.get("usage"):
                        usage = event["usage"]

                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning_content")
                    content = delta.get("content")

                    if (reasoning or content) and ttft_s is None:
                        ttft_s = time.perf_counter() - start
                    if reasoning:
                        yield {"chat_delta": [{"reasoning_content": reasoning}]}
                    if content:
                        yield content
        except (httpx.HTTPError, httpx.StreamError) as exc:
            # Two disjoint hierarchies: transport errors (ReadError etc.) are
            # HTTPError; stream-state errors (StreamClosed) are RuntimeErrors.
            if request_id and request_id in self._cancelled:
                # cancel() closed the stream; the parked read surfaces as a
                # transport or stream-state error depending on where it was
                # blocked. A proxy must never let a client disconnect look
                # like an inference failure: that unloads the registry entry.
                logger.info(
                    f"[{self.load_config.model_name}] request {request_id} stream closed by cancel"
                )
                return
            raise RuntimeError(
                f"[{self.load_config.model_name}] Strata request failed: {exc}"
            ) from exc
        finally:
            if request_id:
                self._active_streams.pop(request_id, None)
                self._cancelled.discard(request_id)

        elapsed = time.perf_counter() - start
        yield self.collect_metrics(
            gen_config,
            usage=usage,
            ttft_s=ttft_s if ttft_s is not None else elapsed,
            elapsed_s=elapsed,
        )

    # --------------------------------------------------------------- metrics

    def collect_metrics(
        self,
        gen_config: OVGenAI_GenConfig,
        usage: Optional[Dict[str, Any]],
        ttft_s: float,
        elapsed_s: float,
    ) -> Dict[str, Any]:
        """OpenArc metrics shape, measured client-side from the proxy.

        'proxy' marks every value as client-measured (httpx timers plus the
        upstream usage object), not engine-reported PerfMetrics.
        """
        usage = usage or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        new_tokens = int(usage.get("completion_tokens") or 0)

        decode_s = max(elapsed_s - ttft_s, 0.0)
        # Non-streaming has no separate decode window (ttft == elapsed): fall
        # back to total elapsed so tpot/throughput are effective values.
        effective_decode_s = decode_s if decode_s > 0 else elapsed_s
        prefill_throughput = round(input_tokens / ttft_s, 2) if ttft_s > 0 else 0
        tpot_ms = (
            round(effective_decode_s / max(new_tokens - 1, 1) * 1000, 5)
            if new_tokens > 0 and effective_decode_s > 0
            else 0
        )
        decode_throughput = (
            round(new_tokens / effective_decode_s, 5)
            if new_tokens > 0 and effective_decode_s > 0
            else 0
        )

        return {
            'load_time (s)': round(self._load_time_s, 2),
            'ttft (s)': round(ttft_s, 2),
            'tpot (ms)': tpot_ms,
            'prefill_throughput (tokens/s)': prefill_throughput,
            'decode_throughput (tokens/s)': decode_throughput,
            'decode_duration (s)': round(decode_s, 5),
            'input_token': input_tokens,
            'new_token': new_tokens,
            'total_token': input_tokens + new_tokens,
            'stream': gen_config.stream,
            'proxy': True,
            'endpoint': self.endpoint,
        }

    # --------------------------------------------------------------- control

    async def cancel(self, request_id: str) -> bool:
        """Cancel an ongoing streaming generation by request_id.

        Closes the active httpx stream; the remote server sees a client
        disconnect and aborts the request.
        """
        resp = self._active_streams.get(request_id)
        if resp is None:
            return False
        self._cancelled.add(request_id)
        try:
            await resp.aclose()
        except Exception as exc:
            # The read side may hold the stream mid-chunk; it will still see
            # the close (or the _cancelled mark) and wind down.
            logger.debug(f"[{self.load_config.model_name}] stream close raised: {exc}")
        logger.info(f"[{self.load_config.model_name}] Cancellation triggered for request {request_id}")
        return True

    async def unload_model(self, registry: ModelRegistry, model_name: str) -> bool:
        """Unregister the model and close the shared HTTP client."""
        removed = await registry.register_unload(model_name)

        if self._client is not None:
            await self._client.aclose()
            self._client = None

        logging.info(f"[{self.load_config.model_name}] unloaded successfully")
        return removed
