# Welcome to OpenARC

[![Discord](https://img.shields.io/discord/1341627368581628004?logo=Discord&logoColor=%23ffffff&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FmaMY7QjG)](https://discord.gg/maMY7QjG)!

Sister repo for Projects using OpenArc: [OpenArcProjects](https://github.com/SearchSavior/OpenArcProjects)

> [!NOTE]
> OpenArc is under active development. Expect breaking changes.

**OpenArc** is an inference engine built with Optimum-Intel to leverage hardware acceleration on Intel CPUs, GPUs and NPUs through OpenVINO runtime that integrates closely with Transformers. 

Under the hood it's a strongly typed fastAPI implementation over a growing collection of Transformers integrated AutoModel classes from Optimum-Intel enabling inference on a wide range of tasks, models and source frameworks.

OpenArc currently supports text generation and text generation with vision. Support for speculative decoding, generating embeddings, speech tasks, image generation, PaddleOCR, and others are planned.
Currently implemented:

[OVModelForCausalLM](https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/modeling_decoder.py#L422)

[OVModelForVisualCausalLM](https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/modeling_visual_language.py#L309)

## Features

- OpenAI compatible endpoints
- Validated OpenWebUI support, but it should work elsewhere
- Load multiple vision/text models concurrently on multiple devices for hotswap.
- **Most** HuggingFace text generation models
- Growing set of vision capable LLMs:
    - Qwen2-VL 
    - Qwen2.5-VL 
    - Gemma3
### Gradio management dashboard
   - Load models with OpenVINO optimizations 
   - Build conversion commands
   - See loaded models
   - Unload models
   - Query detected devices
   - Query device properties
   - View tokenizer data
   - View architecture metadata from config.json
### Performance metrics on every completion
   - ttft: time to generate first token
   - generation_time : time to generate the whole response
   - number of tokens: total generated tokens for that request
   - tokens per second: measures throughput.
   - average token latency: helpful for optimizing zero shot classification tasks
 	  
## System Requirments 

OpenArc has been built on top of the OpenVINO runtime; as a result OpenArc supports the same range of hardware **but requires device specifc drivers** this document will not cover in-depth.

Supported operating system are a bit different for each class of device. Please review [system requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) for OpenVINO 2025.0.0.0 to learn which

- Windows versions are supported
- Linux distributions are supported
- kernel versions are required
	- My system uses version 6.9.4-060904-generic with Ubuntu 24.04 LTS.
- commands for different package managers
- other required dependencies for GPU and NPU

If you need help installing drivers:
	- Join the [Discord](https://discord.gg/PnuTBVcr)
	- Open an issue
	- Use [Linux Drivers](https://github.com/SearchSavior/OpenArc/discussions/11)
	- Use [Windows Drivers](https://github.com/SearchSavior/OpenArc/discussions/12)

<details>
  <summary>CPU</summary>
	
	Intel® Core™ Ultra Series 1 and Series 2 (Windows only )
	
	Intel® Xeon® 6 processor (preview)
	
	Intel Atom® Processor X Series
	    
	Intel Atom® processor with Intel® SSE4.2 support
	
	Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics
	
	6th - 14th generation Intel® Core™ processors
	
	1st - 5th generation Intel® Xeon® Scalable Processors

	ARM CPUs with armv7a and higher, ARM64 CPUs with arm64-v8a and higher, Apple® Mac with Apple silicon

</details>

<details>
  <summary>GPU</summary>
	
    Intel® Arc™ GPU Series

    Intel® HD Graphics

    Intel® UHD Graphics

    Intel® Iris® Pro Graphics

    Intel® Iris® Xe Graphics

    Intel® Iris® Xe Max Graphics

    Intel® Data Center GPU Flex Series

    Intel® Data Center GPU Max Series

</details>

<details>
  <summary>NPU</summary>

    Intel® Core Ultra Series

    This was a bit harder to list out as the system requirments page does not include an itemized list. However, it is safe to assume that if a device contains an Intel NPU it will be supported.

    The Gradio dashboard has tools for querying your device under the Tools tab.

</details>

### Ubuntu

Create the conda environment:

	conda env create -f environment.yaml


Set your API key as an environment variable:

	export OPENARC_API_KEY=<you-know-for-search>

Build Optimum-Intel from source to get the latest support:

```
pip install optimum[openvino]+https://github.com/huggingface/optimum-intel
```

### Windows

1. Install Miniconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)

2. Navigate to the directory containing the environment.yaml file and run

	conda env create -f environment.yaml

Set your API key as an environment variable:

	setx OPENARC_API_KEY=<you-know-for-search>

Build Optimum-Intel from source to get the latest support:

```
pip install optimum[openvino]+https://github.com/huggingface/optimum-intel
```

> [!Tips]
- Avoid setting up the environment from IDE extensions. 
- Try not to use the environment for other ML projects. Soon we will have uv.

## Usage

OpenArc has two components:

- start_server.py - launches the inference server
- start_dashboard.py - launches the dashboard, which manages the server and provides some useful tools


To launch the inference server run

		python start_server.py --host 0.0.0.0 --openarc-port 8000


> host: defines the ip address to bind the server to

> openarc_port: defines the port which can be used to access the server			

To launch the dashboard run

		python start_dashboard.py --openarc-port 8000

> openarc_port: defines the port which requests from the dashboard use

Run these in two different terminals.

> [!NOTE]
> Gradio handles ports natively so the port number does not need to be set. Default is 7860 but it will increment if another instance of gradio is running.

## OpenWebUI

> [!NOTE]
> I'm only going to cover the basics on OpenWebUI here. To learn more and set it up check out the [OpenWebUI docs](https://docs.openwebui.com/).

- From the Connections menu add a new connection
- Enter the server address and port where OpenArc is running **followed by /v1**
Example:
    http://0.0.0.0:8000/v1

- Here you need to set the API key manually
- When you hit the refresh button OpenWebUI sends a GET request to the OpenArc server to get the list of models

In the uvicorn logs where the server is running this request should report:
			
	"GET /v1/models HTTP/1.1" 200 OK

### Usage:

- Load the model you want to use from the dashboard
- Select the connection you just created and use the refresh button to update the list of models

## Convert to [OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html)

There are a few source of models which can be used with OpenArc;

- [OpenVINO LLM Collection on HuggingFace](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd)

- [My HuggingFace repo](https://huggingface.co/Echo9Zulu)
	- My repo contains preconverted models for a variety of architectures and usecases
	- OpenArc supports almost all of them 
  - **These get updated regularly so check back often!**

You can easily craft conversion commands using my HF Space, [Optimum-CLI-Tool_tool](https://huggingface.co/spaces/Echo9Zulu/Optimum-CLI-Tool_tool) or in the OpenArc Dashboard.

This tool respects the positional arguments defined [here](https://huggingface.co/docs/optimum/main/en/intel/openvino/export), then execute commands in the OpenArc environment.

| Models                                                                                                                                      | Compressed Weights |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| [Ministral-3b-instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Ministral-3b-instruct-int4_asym-ov)                                   | 1.85 GB            |
| [Hermes-3-Llama-3.2-3B-awq-ov](https://huggingface.co/Echo9Zulu/Hermes-3-Llama-3.2-3B-awq-ov)							| 1.8 GB |
| [Llama-3.1-Tulu-3-8B-int4_asym-ov](https://huggingface.co/Echo9Zulu/Llama-3.1-Tulu-3-8B-int4_asym-ov/tree/main)                             | 4.68 GB            |
| [Qwen2.5-7B-Instruct-1M-int4-ov](https://huggingface.co/Echo9Zulu/Qwen2.5-7B-Instruct-1M-int4-ov) | 4.46 GB            |
| [Meta-Llama-3.1-8B-SurviveV3-int4_asym-awq-se-wqe-ov](https://huggingface.co/Echo9Zulu/Meta-Llama-3.1-8B-SurviveV3-int4_asym-awq-se-wqe-ov) | 4.68 GB            |
| [Falcon3-10B-Instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Falcon3-10B-Instruct-int4_asym-ov)                                     | 5.74 GB            |
| [Echo9Zulu/phi-4-int4_asym-awq-ov](https://huggingface.co/Echo9Zulu/phi-4-int4_asym-awq-ov) | 8.11 GB            |
| [DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov](https://huggingface.co/Echo9Zulu/DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov/tree/main)             | 7.68 GB            |
| [Phi-4-o1-int4_asym-awq-weight_quantization_error-ov](https://huggingface.co/Echo9Zulu/Phi-4-o1-int4_asym-awq-weight_quantization_error-ov) | 8.11 GB            |
| [Mistral-Small-24B-Instruct-2501-int4_asym-ov](https://huggingface.co/Echo9Zulu/Mistral-Small-24B-Instruct-2501-int4_asym-ov)		| 12.9 GB	     |	
| [Qwen2.5-72B-Instruct-int4-ns-ov](https://huggingface.co/Echo9Zulu/Qwen2.5-72B-Instruct-int4-ns-ov/tree/main)                              | 39 GB              |

Documentation on choosing parameters for conversion is coming soon. 

> [!NOTE]
> The optimum CLI tool integrates several different APIs from several different Intel projects; it is a better alternative than using APIs in from_pretrained() methods. 
> It references prebuilt export configurations for each supported model architecture meaning **not all models are supported** but most are. If you use the CLI tool and get an error about an unsupported architecture follow the link, [open an issue](https://github.com/huggingface/optimum-intel/issues) with references to the model card and the maintainers will get back to you.  

> [!NOTE]
> A naming convention for openvino converted models is coming soon. 

## Performance with OpenVINO runtime


Notes on the test:

- No advanced openvino parameters were chosen
- Fixed input length
- I sent one user message 
- Quant strategies for models are not considered
- I converted each of these models myself (I'm working on standardizing model cards to share this information more directly)
- OpenVINO generates a cache on first inference so metrics are on second generation
- Seconds were used for readability



Test System:

CPU: Xeon W-2255 (10c, 20t) @3.7ghz
GPU: 3x Arc A770 16GB Asrock Phantom
RAM: 128gb DDR4 ECC 2933 mhz
Disk: 4tb ironwolf, 1tb 970 Evo

OS: Ubuntu 24.04
Kernel: 6.9.4-060904-generic

Prompt: "We don't even have a chat template so strap in and let it ride!"

---

### GPU Performance: 1x Arc A770

| Model                                            | Prompt Processing (sec) | Throughput (t/sec) | Duration (sec) | Size (GB) |
| ------------------------------------------------ | ----------------------- | ------------------ | -------------- | --------- |
| Phi-4-mini-instruct-int4_asym-gptq-ov            | 0.41                    | 47.25              | 3.10           | 2.3       |
| Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov         | 0.27                    | 64.18              | 0.98           | 1.8       |
| Llama-3.1-Nemotron-Nano-8B-v1-int4_sym-awq-se-ov | 0.32                    | 47.99              | 2.96           | 4.7       |
| phi-4-int4_asym-awq-se-ov                        | 0.30                    | 25.27              | 5.32           | 8.1       |
| DeepSeek-R1-Distill-Qwen-14B-int4_sym-awq-se-ov  | 0.42                    | 25.23              | 1.56           | 8.4       |
| Mistral-Small-24B-Instruct-2501-int4_asym-ov     | 0.36                    | 18.81              | 7.11           | 12.9      |


### CPU Performance: Xeon W-2255

| Model                                            | Prompt Processing (sec) | Throughput (t/sec) | Duration (sec) | Size (GB) |
| ------------------------------------------------ | ----------------------- | ------------------ | -------------- | --------- |
| Phi-4-mini-instruct-int4_asym-gptq-ov            | 1.02                    | 20.44              | 7.23           | 2.3       |
| Hermes-3-Llama-3.2-3B-int4_sym-awq-se-ov         | 1.06                    | 23.66              | 3.01           | 1.8       |
| Llama-3.1-Nemotron-Nano-8B-v1-int4_sym-awq-se-ov | 2.53                    | 13.22              | 12.14          | 4.7       |
| phi-4-int4_asym-awq-se-ov                        | 4                       | 6.63               | 23.14          | 8.1       |
| DeepSeek-R1-Distill-Qwen-14B-int4_sym-awq-se-ov  | 5.02                    | 7.25               | 11.09          | 8.4       |
| Mistral-Small-24B-Instruct-2501-int4_asym-ov     | 6.88                    | 4.11               | 37.5           | 12.9      |
| Nous-Hermes-2-Mixtral-8x7B-DPO-int4-sym-se-ov    | 15.56                   | 6.67               | 34.60          | 24.2      |


### Resources
---
Learn more about how to leverage your Intel devices for Machine Learning:

[openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

[Inference with Optimum-Intel](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb)

[Optimum-Intel Transformers](https://huggingface.co/docs/optimum/main/en/intel/index)

[NPU Devices](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)

## Acknowledgments

OpenArc stands on the shoulders of several other projects and I appreciate their work. 

[Optimum-Intel](https://github.com/huggingface/optimum-intel)

[OpenVINO](https://github.com/openvinotoolkit/openvino)

[OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)

[Transformers](https://github.com/huggingface/transformers)

[FastAPI](https://github.com/fastapi/fastapi)











