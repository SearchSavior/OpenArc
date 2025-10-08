![CLI Help Screen](assets/openarc_DOOM.png)

[![Discord](https://img.shields.io/discord/1341627368581628004?logo=Discord&logoColor=%23ffffff&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FmaMY7QjG)](https://discord.gg/Bzz9hax9Jq)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Echo9Zulu-yellow)](https://huggingface.co/Echo9Zulu)


> [!NOTE]
> OpenArc is under active development.

**OpenArc** is an inference engine for Intel devices. Serve LLMs, VLMs, Whisper and Kokoro-TTS over OpenAI compatible endpoints.

**OpenArc 2.0** arrives with more endpoints, better UX, pipeline paralell, NPU support and much more!

New Features:
  - Multi GPU Pipeline Paralell
  - CPU offload
  - NPU device 
    - [Help wanted!]().
  - OpenAI compatible Kokoro-TTS OpenVINO implementation 
  - Fully async multi engine, multi task architecture
  - Performance metrics on every request
    - prefill throughput
    - decode throupout 
    - ttft
    - tpot 




## What's new?

Drawing on ideas from llama.cpp, vLLM, transformers and OpenVINO Model Server, Ray and others, OpenArc has evolved into a backyard prediction engine inspired by a cocktail of popular techniques I have had fun building.

### 3 Layers

2.0 can be framed as my answer to one question; "How do you manage a synchronos process in python, asynchronosly?"

Getting there wasn't easy.

















> [!NOTE]
> Interested in contributing? Please open an issue before submitting a PR!

## Features

- OpenAI compatible endpoints
- OpenWebUI support
- Load multiple vision/text models concurrently on multiple devices for hotswap/multi agent workflows
- **Most** HuggingFace text generation models
- Growing set of vision capable LLMs:
    - Qwen2-VL 
    - Qwen2.5-VL 
    - Gemma 3

- Other [multimodal architectures](https://github.com/huggingface/optimum-intel/blob/dd622144bf49333fda5cbce670c841288a46bf16/optimum/intel/openvino/modeling_visual_language.py#L4352) which might work

## Command Line Application
  - Built with click and rich-click
  - OpenArc's server has been thoroughly documented there. Much cleaner!
  - Coupled with officual documentation this makes learning OpenVINO easier. 

### Performance metrics on every completion
   - ttft: time to generate first token
   - generation_time : time to generate the whole response
   - number of tokens: total generated tokens for that request (includes thinking tokens)
   - tokens per second: measures throughput.
   - average token latency: helpful for optimizing zero or few shot tasks
 	  
# OpenArc Command Line Tool

## ```openarc add``` 

Add a model to openarc-config.json for easy loading with ```openarc load```.
> [!NOTE]
> **For vision language models, use `vlm` instead of `llm` for the `--model-type`.**

#### Single device

```
openarc add --model-name <model-name> --model-path <path/to/model> --engine ovgenai --model-type llm --device <target-device>
```

### HETERO device plugin 

See [pipeline-paralellism preview](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview) to learn how to use HETERO device plugin. Some example commands are provided for a few difference scenarios.


#### CPU Offload

```
openarc add --model-name <model-name> -model-path <path/to/model> --engine ovgenai --model-type llm --device <HETERO:GPU.0,CPU> --runtime-config {"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}
```

#### Multi-GPU

```
openarc add --model-name <model-name> --model-path <path/to/model> --engine ovgenai --model-type llm --device <HETERO:GPU.0,GPU.1> --runtime-config {"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}
```

#### Tensor Paralell (CPU only)

```openvino``` Single NUMA node being more efficent due to intra-node communication bandwidth when using more than one socket within a single machine. 

```
openarc add --model-name <model-name> --model-path <path/to/model> --engine ovgenai --model-type llm --device CPU --runtime-config {"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"}
```
---

#### Whisper

```
openarc add --model-name <model-name> --model-path <path/to/whisper> --engine ovgenai --model-type whisper --device <target-device> 
```

#### Kokoro (CPU only)

```
openarc add --model-name <model-name> --model-path <path/to/kokoro> --engine openvino --model-type kokoro --device CPU 
```

### ```runtime-config``` advanced usage

`runtime-config` can accept many options and option combinations to modify ```openvino``` runtime behavior for different scenarios. The devs a 

See OpenVINO documentation on [Inference Optimization](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference.html) to learn more.

#### Some intuition

```openvino``` runtime usually makes decisions in an automatic way based on characteristics of the target device; arguments passed to ```runtime-config``` act as overrides or are injected at compile time, _before_ inference time. 

Documentation on these options can be difficult to understand and are not usually covered in tutorials. Fortunately, `openvino` complains effectively when these are misconfigured, so OpenArc displays these errors in the CLI tool and the server logs to promote experimentation, which is how I learned to apply them.  

## ```openarc list```

Reads added configurations from ```openarc-config.json```.


Display all saved configurations:
```
openarc list
```

Remove a configuration:
```
openarc list --rm --model-name <model-name>
```

## ```openarc serve```

Starts the server.

```
openarc serve start # defauls to 0.0.0.0:8000
```

Configure host and port

```
openarc serve start --host --openarc-port
```

## ```openarc load```

After using ```openarc add``` you can use ```openarc load``` r 

```
openarc load --model-name <model-name>
```

### Errors

OpenArc uses arguments from ```openarc add``` as metadata to make routing decisions; think of it like you are querying for inference code. 

When an ```openarc load``` command fails, the CLI tool displays the full stack trace to help you figure out why.

### Model concurrency

More than one model can be loaded into memory at once in a non blocking way. 

Each model gets its own first in, first out queue, scheduling requests based on when they arrive. Inference of one model can fail without taking down the server, and many inferences can run at once.

However, OpenArc *does not batch requests yet*. Paged attention-like continuous batching for ```llm``` and ```vlm``` will land in a future release.


## ```openarc status```

Calls /openarc/status endpoint and returns a report. Shows loaded models.

```
openarc status
```

![device-detect](assets/openarc_status.png)

## ```openarc tool```

Utility scripts convient way.

To see OpenVINO properties your device supports use:

```
openarc tool device-props
```

To see available devices use

```
openarc tool device-detect
```

![device-detect](assets/cli_tool_device-detect.png)

---

## System Requirments 

- OpenArc has been built on top of the OpenVINO runtime; as a result OpenArc    supports the same range of hardware **but requires device specifc drivers** this document will not cover in-depth.
 
- See [OpenVINO System Requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) to get the most updated information.

> [!NOTE]
> Need help installing drivers? [Join our Discord](https://discord.gg/Bzz9hax9Jq) or open an issue.

## Environment Setup 

<details>
  <summary>Linux</summary>

Install uv from [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

After cloning use:

```
uv sync
```

Activate your environment with:

```
source .venv/bin/activate
```


Set your API key as an environment variable:

	export OPENARC_API_KEY=<you-know-for-search>

Build Optimum-Intel from source to get the latest support:

```
uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
```

</details>

<details>
  <summary>Windows</summary>

1. Install uv from [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

2. Clone OpenArc, enter the directory and run:
  ```
  uv sync
  ```

Set your API key as an environment variable:
```
setx OPENARC_API_KEY openarc-api-key
```
Build Optimum-Intel from source to get the latest support:

```
uv pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel"
```

> [!Tips]
- uv has a [pip interface](https://docs.astral.sh/uv/pip/) as an alternative to the uv interface which is a drop in replacement for pip, but faster. Pretty cool.


</details>

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

## OpenVINO IR and Supported Models

[OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) is an intermediate representation of a model from a variety of source frameworks. Some GGUF

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

- Easily those craft conversion commands using my HF Space, [Optimum-CLI-Tool_tool](https://huggingface.co/spaces/Echo9Zulu/Optimum-CLI-Tool_tool), a Gradio application which helps you GUI-ify an often research intensive process.

- If you use the CLI tool and get an error about an unsupported architecture or "missing export config" follow the link, [open an issue](https://github.com/huggingface/optimum-intel/issues) reference the model card and the maintainers will get back to you.  

Here are some models to get started:

| Models                                                                                                                                      | Compressed Weights |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| [Ministral-3b-instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Ministral-3b-instruct-int4_asym-ov)                                   | 1.85 GB            |
| [Hermes-3-Llama-3.2-3B-awq-ov](https://huggingface.co/Echo9Zulu/Hermes-3-Llama-3.2-3B-awq-ov)							| 1.8 GB |
| [Llama-3.1-Tulu-3-8B-int4_asym-ov](https://huggingface.co/Echo9Zulu/Llama-3.1-Tulu-3-8B-int4_asym-ov/tree/main)                             | 4.68 GB            |
| [DeepSeek-R1-0528-Qwen3-8B-OpenVINO](https://huggingface.co/Echo9Zulu/DeepSeek-R1-0528-Qwen3-8B-OpenVINO) |                |
| [Meta-Llama-3.1-8B-SurviveV3-int4_asym-awq-se-wqe-ov](https://huggingface.co/Echo9Zulu/Meta-Llama-3.1-8B-SurviveV3-int4_asym-awq-se-wqe-ov) | 4.68 GB            |
| [Rocinante-12B-v1.1-int4_sym-awq-se-ov](https://huggingface.co/Echo9Zulu/Rocinante-12B-v1.1-int4_sym-awq-se-ov) | 6.92 GB            |
| [Echo9Zulu/phi-4-int4_asym-awq-ov](https://huggingface.co/Echo9Zulu/phi-4-int4_asym-awq-ov) | 8.11 GB            |
| [DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov](https://huggingface.co/Echo9Zulu/DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov/tree/main)             | 7.68 GB            |
| [Homunculus-OpenVINO](https://huggingface.co/Echo9Zulu/Homunculus-OpenVINO) |             |
| [Mistral-Small-24B-Instruct-2501-int4_asym-ov](https://huggingface.co/Echo9Zulu/Mistral-Small-24B-Instruct-2501-int4_asym-ov)		| 12.9 GB	     |	
| [gemma-3-4b-it-int8_asym-ov](https://huggingface.co/Echo9Zulu/gemma-3-4b-it-int8_asym-ov) | 3.89 GB            |


LLM

VLM

Whisper 

[Echo9Zulu/Kokoro-82M-FP16-OpenVINO](https://huggingface.co/Echo9Zulu/Kokoro-82M-FP16-OpenVINO)




### Resources
---
Learn more about how to leverage your Intel devices for Machine Learning:

[openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

[Inference with Optimum-Intel](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb)

[Optimum-Intel Transformers](https://huggingface.co/docs/optimum/main/en/intel/index)

[NPU Devices](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)

[vllm with IPEX](https://docs.vllm.ai/en/v0.5.1/getting_started/xpu-installation.html)

[Mutli GPU Pipeline Paralell with OpenVINO Model Server](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching_scaling.html#multi-gpu-configuration-loading-models-exceeding-a-single-card-vram)

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










