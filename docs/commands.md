---
icon: lucide/terminal
---

# Commands


After installation run ```openarc --help``` to see focused usage documentation inside the openarc command line tool.

This page contains example commands to help you choose models and configure OpenArc. 

=== "add"

    Writes a model entry to `config.yaml`. Flags are validated against the chosen `--model-type`; a flag that does not apply is rejected, and only flags you pass are written.

    ```
    openarc add \
      --model-name qwen35-08b \
      --model-path /mnt/models/Qwen3.5-0.8B-int8-asym-ov \
      --engine ovgenai \
      --model-type vlm \
      --device CPU
    ```

    `openarc add --help` shows one help panel per `config.yaml` key. See [Configuration](configure.md) for per-model-type examples and the full block reference.

=== "list"

    Reads model entries from `config.yaml`.

    Display all model entries:
    ```
    openarc list
    ```

    Display config metadata for a specific model:
    ```
    openarc list \
      <model-name> \
      -v
    ```

    Remove a model entry:
    ```
    openarc list \
      --remove <model-name>
    ```

=== "serve"

    Starts the server.

    ```
    openarc serve start # defaults to 0.0.0.0:8000
    ```
    Windows 11:

    To exclude _Request failed_ in Windows, you need to specify this address
    ```
    openarc serve start --host 127.0.0.1
    ```

    Configure host and port:

    ```
    openarc serve start \
      --host \
      --port
    ```

    To load models on startup:

    ```
    openarc serve start \
      --load-models model1 model2
    ```

    To require API key authentication:

    ```
    openarc serve start --use-api-key
    ```

    When `--use-api-key` is passed, clients must authenticate with a `Bearer` token matching `OPENARC_API_KEY`. If the environment variable is not set, the server will not start. Without the flag, no authentication is required.

=== "load"

    `openarc load` reads a model's entry from `config.yaml` and loads it onto the OpenArc server.

    OpenArc uses the entry's metadata (engine, model_type, device) to make routing decisions internally; you are querying for correct inference code.

    ```
    openarc load <model-name>
    ```

    To load multiple models at once:

    ```
    openarc load \
      <model-name1> \
      <model-name2> \
      <model-name3>
    ```

    Be mindful of your resources; loading models can be resource intensive! On the first load, OpenVINO performs model compilation for the target `--device`.

    When `openarc load` fails, the CLI tool displays a full stack trace to help you figure out why.

=== "status"

    Calls `/openarc/status` endpoint and returns a report. Shows loaded models.

    ```
    openarc status
    ```

=== "bench"

    Benchmark `llm` performance with pseudo-random input tokens.

    This approach follows [llama-bench](https://github.com/ggml-org/llama.cpp/blob/683fa6ba/tools/llama-bench/llama-bench.cpp#L1922), providing a baseline for the community to assess inference performance between `llama.cpp` backends and `openvino`.

    To support different `llm` tokenizers, we need to standardize how tokens are chosen for benchmark inference. When you set `--p` we select `512` pseudo-random tokens as input_ids from the set of all tokens in the vocabulary.

    `--n` controls the maximum amount of tokens we allow the model to generate; this bypasses `eos` and sets a hard upper limit.

    Default values are:
    ```
    openarc bench \
      <model-name> \
      --p <512> \
      --n <128> \
      --r <5>
    ```

    ![openarc bench](assets/openarc_bench_sample.png)

    `openarc bench` also records metrics in a sqlite database `openarc_bench.db` for easy analysis.

=== "tool"

    Utility scripts.

    To see `openvino` properties your device supports:

    ```
    openarc tool device-props
    ```

    To see available devices:

    ```
    openarc tool device-detect
    ```

    ![device-detect](assets/cli_tool_device-detect.png)
