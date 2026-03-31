# Features

- **NEW!** Containerization with Docker [#60](https://github.com/SearchSavior/OpenArc/issues/60) by @meatposes
- **NEW!** Speculative decoding support for LLMs [#57](https://github.com/SearchSavior/OpenArc/issues/57) by @meatposes
- **NEW!** Streaming cancellation support for LLMs and VLMs
- Multi GPU Pipeline Parallel
- CPU offload/Hybrid device
- NPU device support
- OpenAI compatible endpoints
    - `/v1/models`
    - `/v1/completions`: `llm` only
    - `/v1/chat/completions`
    - `/v1/audio/transcriptions`: `whisper`, `qwen3_asr`
    - `/v1/audio/speech`: `kokoro` only
    - `/v1/embeddings`: `qwen3-embedding` [#33](https://github.com/SearchSavior/OpenArc/issues/33) by @mwrothbe
    - `/v1/rerank`: `qwen3-reranker` [#39](https://github.com/SearchSavior/OpenArc/issues/39) by @mwrothbe
- `jinja` templating with `AutoTokenizers`
- OpenAI Compatible tool calls with streaming and parallel
    - tool call parser currently reads "name", "argument"
- Fully async multi engine, multi task architecture
- Model concurrency: load and infer multiple models at once
- Automatic unload on inference failure
- `llama-bench` style benchmarking for `llm` w/automatic sqlite database
- Metrics on every request
    - ttft
    - prefill_throughput
    - decode_throughput
    - decode_duration
    - tpot
    - load time
    - stream mode
- More OpenVINO [examples](https://github.com/SearchSavior/OpenArc/tree/main/examples)
- OpenVINO implementation of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

!!! note
    Interested in contributing? Please open an issue before submitting a PR!
