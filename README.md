![openarc_DOOM](assets/openarc_DOOM.png)

[![Discord](https://img.shields.io/discord/1341627368581628004?logo=Discord&logoColor=%23ffffff&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FmaMY7QjG)](https://discord.gg/Bzz9hax9Jq)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Echo9Zulu-yellow)](https://huggingface.co/Echo9Zulu)
[![Devices](https://img.shields.io/badge/Devices-CPU%2FGPU%2FNPU-blue)](https://github.com/openvinotoolkit/openvino)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SearchSavior/OpenArc)

> [!NOTE]
> OpenArc is under active development.

**OpenArc** is an inference engine for Intel devices. Serve LLMs, VLMs, Whisper, Kokoro-TTS, Qwen-TTS, Qwen-ASR, Embedding and Reranker models over OpenAI compatible endpoints, powered by OpenVINO on your device. Local, private, open source AI.  

Drawing on ideas from `llama.cpp`, `vLLM`, `transformers`, `OpenVINO Model Server`, `Ray`, `Lemonade`, and other projects cited below, OpenArc has been a way for me to learn about inference engines by trying to build one myself.

Along the way a Discord community has formed around this project! If you are interested in using Intel devices for AI and machine learning, feel free to stop by. 

Thanks to everyone on Discord for their continued support!

> [!NOTE]
> Documentation has been ported to a Zensical site. It's still WIP, and the site isn't live.
> To build and serve the docs after install:
```
zensical serve -a localhost:8004
```
## Table of Contents



- [Features](#features)
- [Quickstart](#quickstart)
  - [Linux](#linux)
  - [Windows](#windows)
  - [Docker](#docker)


## Features
  - NEW! Containerization with Docker #60 by @meatposes
  - NEW! Speculative decoding support for LLMs #57 by @meatposes
  - NEW! Streaming cancellation support for LLMs and VLMs
  - Multi GPU Pipeline Paralell
  - CPU offload/Hybrid device
  - NPU device support
  - OpenAI compatible endpoints
      - `/v1/models`
      - `/v1/completions`: `llm` only
      - `/v1/chat/completions`
      - `/v1/audio/transcriptions`: `whisper`, `qwen3_asr`
      - `/v1/audio/speech`: `kokoro` only       
      - `/v1/embeddings`: `qwen3-embedding` #33 by @mwrothbe
      - `/v1/rerank`: `qwen3-reranker` #39 by @mwrothbe
  - `jinja` templating with `AutoTokenizers`
  - OpenAI Compatible tool calls with streaming and paralell 
    - tool call parser currently reads "name", "argument" 
  - Fully async multi engine, multi task architecture
  - Model concurrency: load and infer multiple models at once
  - Automatic unload on inference failure
  - `llama-bench` style benchmarking for `llm` w/automatic sqlite database
  - metrics on every request
    - ttft
    - prefill_throughput
    - decode_throughput
    - decode_duration
    - tpot
    - load time
    - stream mode
  - More OpenVINO [examples](examples/)
  - OpenVINO implementation of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
  - OpenVINO implementation of Qwen3-TTS and Qwen3-ASR
  

> [!NOTE] 
> Interested in contributing? Please open an issue before submitting a PR!

<div align="right">

[↑ Top](#table-of-contents)

</div>

## Quickstart 

<details id="linux">
<summary><strong style="font-size: 1.2em;">Linux</strong></summary>

<br>

1. OpenVINO requires **device specifc drivers**.
 
- Visit [OpenVINO System Requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) for the latest information on drivers.

2. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

3. After cloning use:

```
uv sync
```

4. Activate your environment with:

```
source .venv/bin/activate
```

Build latest optimum
```
uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
```

Build latest OpenVINO and OpenVINO GenAI from nightly wheels
```
uv pip install --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

5. Set your API key as an environment variable:
```
	export OPENARC_API_KEY=<api-key>
```

6. To get started, run:

```
openarc --help
```

</details>

<details id="windows">
<summary><strong style="font-size: 1.2em;">Windows</strong></summary>

<br>

1. OpenVINO requires **device specifc drivers**.
 
- Visit [OpenVINO System Requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) to get the latest information on drivers.

2. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

3. Clone OpenArc, enter the directory and run:
  ```
  uv sync
  ```

4. Activate your environment with:

```
.venv\Scripts\activate
```

Build latest optimum
```
uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
```

Build latest OpenVINO and OpenVINO GenAI from nightly wheels
```
uv pip install --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

5. Set your API key as an environment variable:
```
setx OPENARC_API_KEY openarc-api-key
```

Or specify a variable in each CMD window:
```
set OPENARC_API_KEY=234 # example
```

6. To get started, run:

```
openarc --help
```

</details>

<details id="docker">
<summary><strong style="font-size: 1.2em;">Docker</strong></summary>

<br>

Instead of fighting with Intel's own docker images, we built our own which is as close to boilerplate as possible. For a primer on docker [check out this video](https://www.youtube.com/watch?v=DQdB7wFEygo).


**Build and run the container:**
```bash
docker-compose up --build -d
```

**Run the container:**
```bash
docker run -d -p 8000:8000 openarc:latest
```
**Enter the container:**
```bash
docker exec -it openarc /bin/bash
```

## Environment Variables

```bash
export OPENARC_API_KEY="openarc-api-key" # default, set it to whatever you want
export OPENARC_AUTOLOAD_MODEL="model_name" # model_name to load on startup
export MODEL_PATH="/path/to/your/models" # mount your models to `/models` inside the container
docker-compose up --build -d
```


Take a look at the [docker-compose](docker-compose.yaml) and [Dockerfile](Dockerfile) ([Battlemage Dockerfile](Battlemage.Dockerfile)) for more details.

</details>

<br>

> [!NOTE]
> Need help installing drivers? [Join our Discord](https://discord.gg/Bzz9hax9Jq) or open an issue.

> [!NOTE] 
> uv has a [pip interface](https://docs.astral.sh/uv/pip/) which is a drop in replacement for pip, but faster. Pretty cool, and a good place to start learning uv.


## Acknowledgments

OpenArc stands on the shoulders of many other projects:

[Optimum-Intel](https://github.com/huggingface/optimum-intel)

[OpenVINO](https://github.com/openvinotoolkit/openvino)

[OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)

[llama.cpp](https://github.com/ggml-org/llama.cpp)

[vLLM](https://github.com/vllm-project/vllm)

[Transformers](https://github.com/huggingface/transformers)

[FastAPI](https://github.com/fastapi/fastapi)

[click](https://github.com/pallets/click)

[rich-click](https://github.com/ewels/rich-click)

```
@article{zhou2024survey,
  title={A Survey on Efficient Inference for Large Language Models},
  author={Zhou, Zixuan and Ning, Xuefei and Hong, Ke and Fu, Tianyu and Xu, Jiaming and Li, Shiyao and Lou, Yuming and Wang, Luning and Yuan, Zhihang and Li, Xiuhong and Yan, Shengen and Dai, Guohao and Zhang, Xiao-Ping and Dong, Yuhan and Wang, Yu},
  journal={arXiv preprint arXiv:2404.14294},
  year={2024}
}
```
Thanks for your work!!










