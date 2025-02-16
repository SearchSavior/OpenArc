# Welcome to OpenARC

**OpenArc** is a lightweight inference API backend for Optimum-Intel from Transformers to leverage hardware acceleration on Intel CPUs, GPUs and NPUs through the OpenVINO runtime using OpenCL drivers.
It has been designed with agentic use cases in mind.

OpenArc serves inference and integrates well with Transformers!

Under the hood it's a strongly typed fastAPI implementation of [OVModelForCausalLM](https://huggingface.co/docs/optimum/main/en/intel/openvino/reference#optimum.intel.OVModelForCausalLM) from Optimum-Intel. So, deploying inference use less of the same code, while reaping the benefits of fastAPI type safety and ease of use. Keep your application code seperate from inference code no matter what hardware configuration has been chosen for deployment.

Here are some features:

- **Strongly typed API with four endpoints**
	- /model/load: loads model and accepts ov_config
	- /model/unload: use gc to purge a loaded model from device memory
	- /generate/text: synchronous execution,  select sampling parameters, token limits : also returns a performance report
	- /status: see the loaded model 
- Each endpoint has a pydantic model keeping exposed parameters easy to maintain or extend.
- Native chat templating

## Usage

OpenArc offers a lightweight approach to decoupling machine learning code from application logic by adding an API layer to serve LLMs using hardware acceleration for Intel devices; in practice OpenArc offers a similar workflow to what's possible with Ollama, LM-Studio or OpenRouter. 

Exposing the _conversation_ parameter from [apply_chat_template](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) method grant's complete control over what get's passed in and out of the model without any intervention required at the template or application level.

	conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) — A list of dicts with “role” and “content” keys, representing the chat history so far.


Use parameters defined in the [Pydanitc models here](https://github.com/SearchSavior/OpenArc/blob/main/src/engine/optimum_inference_core.py) to build a frontend and construct a request body based on inputs from buttons. You can even dump the pydantic models into a prompt and get strong boilerplate for your usecase since each BaseModel directly defines each request type. 

As the AI space moves forward we are seeing all sorts of different paradigms emerge in CoT, agents, etc. From the engineering side of developing systems from a low level, the _conversation_ object seems to be unchanging across both the literature and open source projects. For example, the Deepseek series achieve CoT inside of the same role-based input sequence labels and nearly all training data follows this format. 

For this reason only _conversation_ has been exposed but more options are easy to add.

Notice the typing; if your custom model uses something other than system, user and asssistant roles at inference time you must match the typing to use OpenArc- and that's it!

## Install

A proper .toml will be pushed soon. For now use the provided conda .yaml file;

Make it executable

	sudo chmod +x environment.yaml
 
Then create the conda environment

	conda env create -f environment.yaml


## Workflow

- Either download or convert an LLM
- Use the /model/load endpoint
- Use the /generate/text endpoint for inference
- Manage the conversation dictionary in code somewhere else. 

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
Learn more about model conversion and working with Intel Devices generally in the openvino_utilities notebook.


NOTE: The optimum CLI tool integrates several different APIs from several different Intel projects; it is a better alternative than using APIs in the from_pretrained() methods. Also, it references prebuilt configurations for each supported model architecture meaning that **not all models are natively supported** but most are. If you use the CLI tool and get an error about an unsupported architecture follow the link, open an issue with references to the model card and the maintainers will get back to you.  

## Known Issues

- Right now the /generate/text endpoint can return the whole conversation dict. 

## Roadmap

- Define a pyprojectoml o
- More documentation about how to use the various openvino parameters like ENABLE_HYPERTHREADING and INFERENCE_NUM_THREADS for CPU only which are not included in this release yet.


- Add an openai proxy
- Add support for vision models like Qwen2-VL since CPU performance is really good. Benchmarks coming!
- Add support for querying tokenizers

- Add support for containerized deployments and documentation for adding devices to docker-compose


### Resources
---
Learn more about how to leverage your Intel devices for Machine Learning:

[openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

[Inference with Optimum-Intel](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb)

### Other Resources

[smolagents](https://huggingface.co/blog/smolagents)- [CodeAgent](https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py#L1155) is especially interesting.








