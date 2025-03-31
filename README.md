# Welcome to OpenARC

NEW: OpenWebUI is now supported!

NEW: Join the offical [![Discord](https://img.shields.io/discord/1341627368581628004?logo=Discord&logoColor=%23ffffff&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FmaMY7QjG)](https://discord.gg/maMY7QjG)!

NEW: Sister repo for Projects using OpenArc: [OpenArcProjects](https://github.com/SearchSavior/OpenArcProjects)

> [!NOTE]
> OpenArc is under active development. Expect breaking changes.


**OpenArc** is a lightweight inference API backend for Optimum-Intel from Transformers to leverage hardware acceleration on Intel CPUs, GPUs and NPUs through the OpenVINO runtime.

OpenArc serves inference for LLMs supported by Optimum-Intel including text generation, text generation with vision. There are plans to expand support for speculative decoding, generating embeddings, speech tasks, image generation and more.

Under the hood it's a strongly typed fastAPI implementation over a growing collection of Transformers integrated AutoModel classes enabling inference on a wide range of models. For now text generation and text generation with vision are  So, deploying inference uses less of the same code, while reaping the benefits of hardware acceleration on Intel devices. Keep application logic separate from inference code no matter what hardware configuration has been chosen for deployment.

Here are some features:

- **Strongly typed API**
	- optimum/model/load: loads model and accepts ov_config
	- optimum/model/unload: use gc to purge a loaded model from device memory
	- optimum/status: see the loaded model 

- **OpenAI-compatible endpoints**
	- /v1/chat/completions: implementation of the OpenAI API for chat completions
	- /v1/models

	Validated with:
	- OpenWebUI (!)

- **Gradio Dashboard**
	- A dashboard for loading models and interacting with OpenArc's API
	- Tools for querying device properties
	- GUI for building model conversion commands 
	- Query tokenizers and model architecture


## Design Philosophy: Conversation as the Atomic Unit of LLM Programming

OpenArc offers a lightweight approach to decoupling inference code from application logic by adding an API layer to serve machine learning models using hardware acceleration for Intel devices; in practice OpenArc offers a similar workflow to what's possible with Ollama, LM-Studio or OpenRouter. 

As the AI space moves forward we are seeing all sorts of different paradigms emerge in CoT, agents, etc. Every design pattern using LLMs converges to some manipulation of the chat sequence stored in _conversation_ or (outside of transformers)_messages_. No matter what you build, you'll need to manipulate the chat sequence in some way that has nothing to do with inference. By the time data has been added to _conversation_ all linear algebra, tokenization and decoding has been taken care of. OpenArc embraces this distinction.

Exposing _conversation_ from [apply_chat_template](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) enables complete control over what get's passed in and out of the model without any intervention required at the template level. Projects using OpenArc can focus on application logic and less on inference code.  Check out the typing for _conversation_:

	conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) — A list of dicts with “role” and “content” keys, representing the chat history so far.

Match the typing, manage the logic for assigning roles, update the sequence with content and you're set to build whatever you want using Intel CPUs, GPUs, and NPUs for acceleration.

To poke around with this approach, skip the OpenAI API and dive right into the [requests](https://github.com/SearchSavior/OpenArc/tree/main/scripts/requests) examples which work well inside of coding prompts. Treat these like condensed documentation for the OpenArc API to get you started.

Only _conversation_ has been exposed for now. There are two other useful options; _tools_ and _documents_ which will be added in future releases- these are much harder to test ad hoc and require knowing model-specifc facts about training, manually mapping tools to tokens and building those tools. Each of these wrap RAG documents and tool calls in special tokens which should increase reliability for structured outputs at a lower level of abstraction; instead of using the prompt to tell the model what context to use the tokens do part ofthis work for us. OpenArc will not define some class to use for mapping tools to tokens, instead it empowers developers to define their own tools and documents with an engine tooled to accept them as part of a request.

## System Requirments 

OpenArc has been built on top of the OpenVINO runtime; as a result OpenArc supports the same range of hardware.

Supported operating system are a bit different for each class of device. Please review [system requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) for OpenVINO 2025.0.0.0 to learn which

- Windows versions are supported
- Linux distributions are supported
- kernel versions are requiered
	- My system uses version 6.9.4-060904-generic with Ubuntu 24.04 LTS.
	- This matters more for GPU and NPU
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

## Install

This documentation will not provide guidance for setting up device drivers. Instead, I will try to help on a case-by-case basis. 

Either open an issue or use dedicated discussions for your OS.

Windows discussion lives [here](https://github.com/SearchSavior/OpenArc/discussions/12).

Linux discussion lives [here](https://github.com/SearchSavior/OpenArc/discussions/11).


> Come prepared with details about your system, errors you are experiencing and the steps you've taken so far to resolve the issue. These discussions offer an opportunity to supplement official documentation so you should assume someone in the future can use your help.

Feel free to use the Discord for this as well but be prepared to document your solution on GitHub if asked! 

### Ubuntu

Create the conda environment:

	conda env create -f environment.yaml


Set your API key as an environment variable:

	export OPENARC_API_KEY=<you-know-for-search>


### Windows

1. Install Miniconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)

2. Navigate to the directory containing the environment.yaml file and run

	conda env create -f environment.yaml

Set your API key as an environment variable:

	setx OPENARC_API_KEY=<you-know-for-search>

> [!Tips]
- Avoid setting up the environment from IDE extensions. 
- DO NOT USE THE ENVIRONMENT FOR ANYTHING ELSE. Soon we will have uv.

## Usage

OpenArc has two components:

- start_server.py - launches the inference server
- start_dashboard.py - launches the dashboard, which manages the server


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
- Enter the server address and port where OpenArc is running
- Here you need to set the API key manually
- When you hit the refresh button OpenWebUI sends a GET request to the OpenArc server to get the list of models

In the uvicorn logs where the server is running this request should report:
			
	"GET /v1/models HTTP/1.1" 200 OK

### Usage:

- Load the model you want to use from the dashboard
- Select the connection you just created and use the refresch button to update the list of models


#### - You can now use an OpenVINO accelerated model in a chat with OpenWebUI community tooling! This is a big step forward for OpenArc. Getting this right so early in the project positions OpenArc to be mature nicely



Some notes about the current version:

- To see performance metrics read the logs displayed where start_server.py is running.
- I *strongly* reccomend using these together but they are independent by design.
- The dashboard has documentation and tools intended to help you get started working with both OpenVINO and Intel hardware.

## Convert to [OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html)

There are a few source of models which can be used with OpenArc;

- [OpenVINO LLM Collection on HuggingFace](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd)

- [My HuggingFace repo](https://huggingface.co/Echo9Zulu)
	- My repo contains preconverted models for a variety of architectures and usecases
	- OpenArc supports almost all of them 

You can easily craft conversion commands using my HF Space, [Optimum-CLI-Tool_tool](https://huggingface.co/spaces/Echo9Zulu/Optimum-CLI-Tool_tool) or in the OpenArc Dashboard.

This tool respects the positional arguments defined [here](https://huggingface.co/docs/optimum/main/en/intel/openvino/export), then execute commands in the OpenArc environment.

 Benchmarks for CPU, GPU and code examples are coming soon.

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



## Design Philosophy: Conversation as the Atomic Unit of LLM Programming

OpenArc offers a lightweight approach to decoupling inference code from application logic by adding an API layer to serve machine learning models using hardware acceleration for Intel devices; in practice OpenArc offers a similar workflow to what's possible with Ollama, LM-Studio or OpenRouter. 

As the AI space moves forward we are seeing all sorts of different paradigms emerge in CoT, agents, etc. Every design pattern using LLMs converges to some manipulation of the chat sequence stored in _conversation_ or (outside of transformers)_messages_. No matter what you build, you'll need to manipulate the chat sequence in some way that has nothing to do with inference. By the time data has been added to _conversation_ all linear algebra, tokenization and decoding has been taken care of. OpenArc embraces this distinction.

Exposing _conversation_ from [apply_chat_template](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) enables complete control over what get's passed in and out of the model without any intervention required at the template level. Projects using OpenArc can focus on application logic and less on inference code.  Check out the typing for _conversation_:

	conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) — A list of dicts with “role” and “content” keys, representing the chat history so far.

Match the typing, manage the logic for assigning roles, update the sequence with content and you're set to build whatever you want using Intel CPUs, GPUs, and NPUs for acceleration.

To poke around with this approach, skip the OpenAI API and dive right into the [requests](https://github.com/SearchSavior/OpenArc/tree/main/scripts/requests) examples which work well inside of coding prompts. Treat these like condensed documentation for the OpenArc API to get you started.

Only _conversation_ has been exposed for now. There are two other useful options; _tools_ and _documents_ which will be added in future releases- these are much harder to test ad hoc and require knowing model-specifc facts about training, manually mapping tools to tokens and building those tools. Each of these wrap RAG documents and tool calls in special tokens which should increase reliability for structured outputs at a lower level of abstraction; instead of using the prompt to tell the model what context to use the tokens do part ofthis work for us. OpenArc will not define some class to use for mapping tools to tokens, instead it empowers developers to define their own tools and documents with an engine tooled to accept them as part of a request.







## Planned Features




### Resources
---
Learn more about how to leverage your Intel devices for Machine Learning:

[openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

[Inference with Optimum-Intel](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb)

### Other Resources

[smolagents](https://huggingface.co/blog/smolagents)- [CodeAgent](https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py#L1155) is especially interesting.










