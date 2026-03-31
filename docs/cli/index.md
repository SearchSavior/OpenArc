# OpenArc CLI

OpenArc command line tool helps you manage the server by packaging requests; every operation the command line does can be scripted programmatically, but using the command tool will help you get a feel for what the server does and how you can use it.

## Getting Started

After choosing a model, use commands in this order:

1. Add model configurations with [`openarc add`](add.md)
2. Show added configurations with [`openarc list`](list.md)
3. Launch the server with [`openarc serve`](serve.md)
4. Load models with [`openarc load`](load.md)
5. Check a model's status using [`openarc status`](status.md)
6. Benchmark performance with [`openarc bench`](bench.md)
7. Call utility scripts with [`openarc tool`](tool.md)

Here's an example for Gemma 3 VLM on GPU:

```
openarc add --model-name <model-name> --model-path <path/to/model> --engine ovgenai --model-type vlm --device GPU.0 --vlm-type gemma3
```

And an LLM on GPU:

```
openarc add --model-name <model-name> --model-path <path/to/model> --engine ovgenai --model-type llm --device GPU.0
```

Qwen3 ASR example:

```
openarc add --model-name qwen3_asr --model-path <path/to/qwen3_asr_ir> --engine openvino --model-type qwen3_asr --device CPU
python demos/qwen3_asr_transcribe.py <path/to/audio.wav> --model qwen3_asr
```

Each command has groups of options which offer fine-grained control of both server behavior and performance optimizations. Use `openarc [OPTION] --help` to see available arguments at any time.
