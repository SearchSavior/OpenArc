# Welcome to OpenARC

**OpenArc** is a lightweight inference API backend for Optimum-Intel from Transformers to leverage hardware acceleration on Intel CPUs, GPUs and NPUs through the OpenVINO runtime.
It has been designed with agentic and chat use cases in mind. 

OpenArc serves inference and integrates well with Transformers!

Under the hood it's a strongly typed fastAPI implementation of [OVModelForCausalLM](https://huggingface.co/docs/optimum/main/en/intel/openvino/reference#optimum.intel.OVModelForCausalLM) from Optimum-Intel. So, deploying inference use less of the same code, while reaping the benefits of fastAPI type safety and ease of use. Keep your application code separate from inference code no matter what hardware configuration has been chosen for deployment.

Here are some features:

- **Strongly typed API with openai-compatible endpoints**
	- optimum/model/load: loads model and accepts ov_config
	- optimum/model/unload: use gc to purge a loaded model from device memory
	- optimum/generate: generate text with sampling, temp
	- optimum/status: see the loaded model 

- **OpenAI-compatible endpoints**
	- /v1/chat/completions: a minimal implementation of the OpenAI API for chat completions
	- /v1/models

	Tested with:
	- Openwebui
	- SillyTavern

- Each endpoint has a pydantic model keeping exposed parameters easy to maintain or extend.
- Native chat templating

- **Gradio Dashboard**
	- A dashboard for loading models and interacting with OpenArc's API
	- Tools for querying device properties
	- GUI model conversion
	- Querying tokenizers and model architecture
	- A chat interface for interacting with the loaded model
		The expection is to use this for development but it works well for testing out the acceleration benefits of OpenArc.


## Workflow

- Either download or convert an LLM to OpenVINO IR
- Load the model using the /optimum/model/load endpoint OR use the Gradio dashboard
- Manage the conversation dictionary in code somewhere else. 

## Conversation as the Atomic Unit of LLM Programming

OpenArc offers a lightweight approach to decoupling machine learning code from application logic by adding an API layer to serve LLMs using hardware acceleration for Intel devices; in practice OpenArc offers a similar workflow to what's possible with Ollama, LM-Studio or OpenRouter. 

As the AI space moves forward we are seeing all sorts of different paradigms emerge in CoT, agents, etc. In my work I noticed a design pattern; programming with LLMs converges to manipulating the _conversation_ object, a data structure which stores the chat sequence.  Every design pattern using LLMsconverges to some manipulation of the chat sequence stored in _conversation_ or (outside of transformers)_messages_. No matter what you build, you'll need to manipulate the chat sequence in some way.

Managing the _conversation_ has nothing to do with inference; by the time data has been added to _conversation_ all the linear algebra and tokenization has been taken care of. 

Exposing the _conversation_ parameter from [apply_chat_template](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) method grant's complete control over what get's passed in and out of the model without any intervention required at the template level; once you start manipulating the _conversation_ object you inherit different challenges.

	conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) — A list of dicts with “role” and “content” keys, representing the chat history so far.

Notice the typing; if your custom model uses something other than system, user and asssistant roles at inference time you must match the typing to use OpenArc- and that's it!


Use parameters defined in the [Pydanitc models here](https://github.com/SearchSavior/OpenArc/blob/main/src/engine/optimum_inference_core.py) to build a frontend or construct a request body based on inputs from buttons.

For example, the Deepseek series achieve CoT inside of the same role-based input sequence labels. As of Feb 2025, reasoning happens inside of <think> tags which themsleves are part of the assistant role. Features which display"thoughts" fail to trigger in open source frontends- no <think> tag was generated but we still get CoT as part of the assistant role content. Moreover, _conversation_ is inherited from Transformers; as SOTA advances we can expect it to enrich the feature set of OpenArc.

Only _conversation_ has been exposed for now. There are two other options; _tools_ and _documents_ which will be added in future releases- these are much harder to test ad hoc and require knowing model-specifc facts about training, manually mapping tools to tokens, building those tools etc.

Notice the typing; if your custom model uses something other than system, user and asssistant roles at inference time you must match the typing to use OpenArc- and that's it!

## Use cases

## System Requirments 

OpenArc has been built on top of the OpenVINO runtime; as a result OpenArc supports the same range of hardware.

Operating system support are a bit different for each class of device. Please review [system requirments](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html#cpu) for OpenVINO 2025.0.0.0 to learn which

- Linux distributions are supported
- kernel versions
- commands for different packacge managers
- other required dependencies  

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

Note: Do not expect zero-shot guidance. Come prepared with details about your system, errors you are experiencing and the steps you've taken so far to resolve the issue. These discussions are offer an opportunity to supplement official documentation so you should assume someone in the future needs your help.

Feel free to use the Discord for this as well but be prepared to document your solution. 

### Ubuntu

Make it executable

	sudo chmod +x environment.yaml
 
Then create the conda environment

	conda env create -f environment.yaml


### Windows

1. Install Miniconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)

2. Navigate to the directory containing the environment.yaml file and run

	conda env create -f environment.yaml

### Tips

- Avoid setting up the environment from IDE extensions. 
- DO NOT USE THE ENVIRONMENT FOR ANYTHING ELSE. Soon we will have Poetry to enforce; until then expect breaking changes for anything which Transformers depends on.
- If you are struggling 



### Convert to [OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html)

Find over 40+ preconverted text generation models of larger variety that what's offered by the official repos from [Intel](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd) on my [HuggingFace](https://huggingface.co/Echo9Zulu).

You can easily craft conversion commands using my HF Space, [Optimum-CLI-Tool_tool](https://huggingface.co/spaces/Echo9Zulu/Optimum-CLI-Tool_tool)! 

This tool respects the positional arguments defined [here](https://huggingface.co/docs/optimum/main/en/intel/openvino/export) and is meant to be used locally in your OpenArc environment.

 
| Models                                                                                                                                      | Compressed Weights |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| [Ministral-3b-instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Ministral-3b-instruct-int4_asym-ov)                                   | 1.85 GB            |
| [Hermes-3-Llama-3.2-3B-awq-ov](https://huggingface.co/Echo9Zulu/Hermes-3-Llama-3.2-3B-awq-ov)							| 1.8 GB |
[Llama-3.1-Tulu-3-8B-int4_asym-ov](https://huggingface.co/Echo9Zulu/Llama-3.1-Tulu-3-8B-int4_asym-ov/tree/main)                             | 4.68 GB            |
| [Falcon3-10B-Instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Falcon3-10B-Instruct-int4_asym-ov)                                     | 5.74 GB            |
| [DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov](https://huggingface.co/Echo9Zulu/DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov/tree/main)             | 7.68 GB            |
| [Phi-4-o1-int4_asym-awq-weight_quantization_error-ov](https://huggingface.co/Echo9Zulu/Phi-4-o1-int4_asym-awq-weight_quantization_error-ov) | 8.11 GB            |
| [Mistral-Small-24B-Instruct-2501-int4_asym-ov](https://huggingface.co/Echo9Zulu/Mistral-Small-24B-Instruct-2501-int4_asym-ov)		| 12.9 GB	     |	
| [Qwen2.5-72B-Instruct-int4-ns-ov](https://huggingface.co/Echo9Zulu/Qwen2.5-72B-Instruct-int4-ns-ov/tree/main))                              | 39 GB              |

Additionally, 
Learn more about model conversion and working with Intel Devices in the [openvino_utilities](https://github.com/SearchSavior/OpenArc/blob/main/docs/openvino_utils.ipynb) notebook.


NOTE: The optimum CLI tool integrates several different APIs from several different Intel projects; it is a better alternative than using APIs in from_pretrained() methods. Also, it references prebuilt configurations for each supported model architecture meaning that **not all models are supported** but most are. If you use the CLI tool and get an error about an unsupported architecture follow the link, open an issue with references to the model card and the maintainers will get back to you.  

## Known Issues

- Streaming does not return performance metrics

## Planned Features

- Define a pyprojectoml
- Improve OpenAI API compatibility for different tooling
- Add benchmarking tools
- More documentation about how to use ov_config
- Add robust feature support for Qwen2-VL from [OVModelForVisualCausalLM](https://github.com/huggingface/optimum-intel/blob/c9ff040327bda796458d7f105979be3665431f1c/optimum/intel/openvino/modeling_visual_language.py#L287)
- Add support for loading multiple models into memory and on different devices
- Add docker-compose examples



### Resources
---
Learn more about how to leverage your Intel devices for Machine Learning:

[openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

[Inference with Optimum-Intel](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb)

### Other Resources

[smolagents](https://huggingface.co/blog/smolagents)- [CodeAgent](https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py#L1155) is especially interesting.








