![openarc_DOOM](assets/openarc_DOOM.png)

[![Discord](https://img.shields.io/discord/1341627368581628004?logo=Discord&logoColor=%23ffffff&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FmaMY7QjG)](https://discord.gg/Bzz9hax9Jq)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Echo9Zulu-yellow)](https://huggingface.co/Echo9Zulu)


> [!]
> OpenArc is under active development.

**OpenArc** is an inference engine for Intel devices. Serve LLMs, VLMs, Whisper, Kokoro-TTS and Embedding models over OpenAI compatible endpoints, powered by OpenVINO.

**OpenArc 2.0** arrives with more endpoints, better UX, pipeline paralell, NPU support and much more!

## What's new?

Drawing on ideas from llama.cpp, vLLM, transformers, OpenVINO Model Server, Ray, Lemonade and other projects, OpenArc has evolved into a capable serving engine for AI workloads on Intel devices.

New Features:
  - Multi GPU Pipeline Paralell
  - CPU offload/Hybrid device
  - NPU device support
  - OpenAI compatible endpoints
      - `/v1/models`
      - `/v1/chat/completions`: tool use, streaming
      - `/v1/embeddings`: 
      - `/v1/audio/transcriptions`: Whisper only
      - `/v1/audio/speech`: Kokoro only       
  - `jinja` templating with AutoTokenizers
  - Fully async multi engine, multi task architecture
  - Model concurrency: load and infer multiple models at once
  - Performance metrics on every request
    - prefill_throughput
    - ttft
    - decode_throughput
    - decode_duration
    - tpot
    - load time
    - stream mode
  - More

> [!] Interested in contributing? Please open an issue before submitting a PR!

## Quickstart 

<details>
  <summary>Linux</summary>
<br>

1. OpenVINO requires **device specifc drivers**.
 
- Visit [OpenVINO System Requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) for the latest information on drivers.

> [!] Need help installing drivers? [Join our Discord](https://discord.gg/Bzz9hax9Jq) or open an issue.

2. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

3. After cloning use:

```
uv sync
```

4. Activate your environment with:

```
source .venv/bin/activate
```


5. Set your API key as an environment variable:

	export OPENARC_API_KEY=<api-key>

> [!] uv has a [pip interface](https://docs.astral.sh/uv/pip/) which is a drop in replacement for pip, but faster. Pretty cool, and a good place to start.


6. To get started, run:

```
openarc --help
```

</details>

<details>
  <summary>Windows</summary>

1. OpenVINO requires **device specifc drivers**.
 
- Visit [OpenVINO System Requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) to get the latest information on drivers.

> [!]
> Need help installing drivers? [Join our Discord](https://discord.gg/Bzz9hax9Jq) or open an issue.


2. Install uv from [astral](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

3. Clone OpenArc, enter the directory and run:
  ```
  uv sync
  ```

4. Set your API key as an environment variable:
```
setx OPENARC_API_KEY openarc-api-key
```

5. To get started, run:

```
openarc --help
```

> [!] uv has a [pip interface](https://docs.astral.sh/uv/pip/) which is a drop in replacement for pip, but faster. Pretty cool, and a good place to start.

</details>

## OpenArc CLI

<details>
  <summary><code>openarc add</code></summary>
<br>

Add a model to openarc-config.json for easy loading with ```openarc load```.
> [!] For vision language models, use `vlm` instead of `llm` for `--model-type`.

#### Single device

```
openarc add --model-name <model-name> --model-path <path/to/model> --engine <engine> --model-type <model-type> --device <target-device>
```

#### Whisper

```
openarc add --model-name <model-name> --model-path <path/to/whisper> --engine ovgenai --model-type whisper --device <target-device> 
```

#### Kokoro (CPU only)

```
openarc add --model-name <model-name> --model-path <path/to/kokoro> --engine openvino --model-type kokoro --device CPU 
```


#### ```runtime-config```

Accepts many options to modify ```openvino``` runtime behavior for different inference scenarios. OpenArc will report errors to the server when the fail, making experimentation easy.

See OpenVINO documentation on [Inference Optimization](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference.html) to learn more about what can be customized. 

Review [pipeline-paralellism preview](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview) to learn how you can customize multi device inference using HETERO device plugin. Some example commands are provided for a few difference scenarios:

#### Hybrid Mode/CPU Offload

```
openarc add --model-name <model-name> -model-path <path/to/model> --engine ovgenai --model-type llm --device <HETERO:GPU.0,CPU> --runtime-config {"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}
```

#### Multi-GPU Pipeline Paralell

```
openarc add --model-name <model-name> --model-path <path/to/model> --engine ovgenai --model-type llm --device <HETERO:GPU.0,GPU.1> --runtime-config {"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}
```

#### Tensor Paralell (CPU only)

Requires more than one CPU socket in a single node.

```
openarc add --model-name <model-name> --model-path <path/to/model> --engine ovgenai --model-type llm --device CPU --runtime-config {"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"}
```
---

</details>

<details>
  <summary><code>openarc list</code></summary>
<br>

Reads added configurations from ```openarc-config.json```.

Display all saved configurations:
```
openarc list
```

Remove a configuration:
```
openarc list --rm --model-name <model-name>
```

</details>

<details>
  <summary><code>openarc serve</code></summary>
<br>


Starts the server.

```
openarc serve start # defauls to 0.0.0.0:8000
```

Configure host and port

```
openarc serve start --host --openarc-port
```

</details>

<details>
  <summary><code>openarc load</code></summary>
<br>


After using ```openarc add``` you can use ```openarc load``` to read the added configuration and load the model onto the OpenArc server. 

```
openarc load --model-name <model-name>
```

OpenArc uses arguments from ```openarc add``` as metadata to make routing decisions internally; you are querying for inference code. 

When an ```openarc load``` command fails, the CLI tool displays the full stack trace to help you figure out why.

</details>

<details>
  <summary><code>openarc status</code></summary>
<br>


Calls /openarc/status endpoint and returns a report. Shows loaded models.

```
openarc status
```

![device-detect](assets/openarc_status.png)

</details>

<details>
  <summary><code>openarc tool</code></summary>
<br>


Utility scripts.

To see OpenVINO properties your device supports use:

```
openarc tool device-props
```

To see available devices use:

```
openarc tool device-detect
```

![device-detect](assets/cli_tool_device-detect.png)

---

</details>




## Supported Models

[OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) is an intermediate representation of a model from a variety of source frameworks.

Based on the tasks we support models usually come from Pytorch, 

There are a few sources of preconverted models which can be used with OpenArc;

- [OpenVINO LLM Collection on HuggingFace](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd)

- [My HuggingFace repo](https://huggingface.co/Echo9Zulu)
	- My repo contains preconverted models for a variety of architectures and usecases
	- OpenArc supports almost all of them 
  - Includes NSFW, ERP and "exotic" community finetunes that Intel doesn't host take advantage!
  - **These get updated regularly so check back often!**
  - If you read this here, *mention it on Discord* and I can quant a model you want to try. 

- [Optimum-CLI Conversion documentation](https://huggingface.co/docs/optimum/main/en/intel/openvino/export) to learn how you can convert models to OpenVINO IR from other formats.

- [Supported Architectures](https://huggingface.co/docs/optimum/main/en/intel/openvino/models)List of models which can be converted to OpenVINO IR

- [Optimum-CLI-Tool_tool](https://huggingface.co/spaces/Echo9Zulu/Optimum-CLI-Tool_tool), a Gradio application which helps you GUI-ify an often research intensive process.

- If you use the CLI tool and get an error about an unsupported architecture or "missing export config" follow the link, [open an issue](https://github.com/huggingface/optimum-intel/issues) reference the model card and the maintainers will get back to you.  

Here are some models to get started:

| Models                                                                           LLMs 

| [Qwen3-4B-Instruct-2507-int4_asym-awq-ov](https://huggingface.co/Echo9Zulu/Qwen3-4B-Instruct-2507-int4_asym-awq-ov) | 
                             | Compressed Weights |
| [Magistral-Small-2509-Text-Only-int4_asym-awq-ov](https://huggingface.co/Echo9Zulu/Magistral-Small-2509-Text-Only-int4_asym-awq-ov)   |	
| [Hermes-4-70B-int4_asym-awq-ov](https://huggingface.co/Echo9Zulu/Hermes-4-70B-int4_asym-awq-ov) | 
| 


VLMs

[gemma-3-4b-it-int8_asym-ov](https://huggingface.co/Echo9Zulu/gemma-3-4b-it-int8_asym-ov) | 3.89 GB            |





Whisper 

[distil-whisper-large-v3-int8-ov](https://huggingface.co/OpenVINO/distil-whisper-large-v3-int8-ov)

Kokoro

[Echo9Zulu/Kokoro-82M-FP16-OpenVINO](https://huggingface.co/Echo9Zulu/Kokoro-82M-FP16-OpenVINO)

Embedding

[Qwen3-Embedding-0.6B-int8_asym-ov](https://huggingface.co/Echo9Zulu/Qwen3-Embedding-0.6B-int8_asym-ov)

### Learning Resources
---
Learn more about how to leverage your Intel devices for Machine Learning:

[openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

[Inference with Optimum-Intel](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb)

[Optimum-Intel Transformers](https://huggingface.co/docs/optimum/main/en/intel/index)

[NPU Devices](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)

[vllm with IPEX](https://docs.vllm.ai/en/v0.5.1/getting_started/xpu-installation.html)

[Mutli GPU Pipeline Paralell with OpenVINO Model Server](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching_scaling.html#multi-gpu-configuration-loading-models-exceeding-a-single-card-vram)





## OpenWebUI

> [!NOTE]
> I'm only going to cover the basics on OpenWebUI here. To learn more and set it up check out the [OpenWebUI docs](https://docs.openwebui.com/).

- From the Connections menu add a new connection
- Enter the server address and port where OpenArc is running **followed by /v1**
Example:
    http://0.0.0.0:8000/v1

- Here you need to set the API key manually
- When you hit the refresh button OpenWebUI sends a GET request to the OpenArc server to get the list of models at v1/models

Serverside logs should report:
			
	"GET /v1/models HTTP/1.1" 200 OK 

### Usage:

- Load the model you want to use from openarc cli
- Select the connection you just created and use the refresh button to update the list of models
- if you use API keys and have a list of models these might be towards the bottom

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

Thank for your work!!










