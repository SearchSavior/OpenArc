---
icon: lucide/terminal
---

# Commands


After installation run ```openarc --help``` to see focused usage documentation inside the openarc command line tool.

This page contains example commands to help you choose models and configure OpenArc. 

=== "add"

    Add a model to `openarc_config.json` for easy loading with `openarc load`.

    === "Single device"

        ```
        openarc add \
          --model-name <model-name> \
          --model-path <path/to/model> \
          --engine <engine> \
          --model-type <model-type> \
          --device <target-device>
        ```

        To see what options you have for `--device`, use `openarc tool device-detect`.

    === "VLM"

        ```
        openarc add \
          --model-name <model-name> \
          --model-path <path/to/model> \
          --engine <engine> \
          --model-type <model-type> \
          --device <target-device> \
          --vlm-type <vlm-type>
        ```

        Getting VLM to work the way I wanted required using VLMPipeline in ways that are not well documented. You can look at the [code](src/engine/ov_genai/vlm.py#L33) to see how OpenArc's VLM backend passes images. Basically, it involves slicing the input sequence by scanning for when there's in image and injecting appropriate tokens. Honestly I have no ideas why they built VLMPipeline this way, but to support all the architectures my approach was easier in the end.

        `vlm-type` maps a vision token for a given architecture. Use `openarc add --help` to see the available options. The server will complain if you get anything wrong, so it should be easy to figure out.


        NOTE: you don't have to pass `Vision Token`; these are mapped to the `vlm-type` `openarc add` argument so use that instead. 

        | `--vlm-type`   | Vision token                                        |
        |----------------|-----------------------------------------------------|
        | `internvl2`    | `<image>`                                           |
        | `llava15`      | `<image>`                                           |
        | `llavanext`    | `<image>`                                           |
        | `minicpmv26`   | `(<image>./</image>)`                               |
        | `phi3vision`   | `<\|image_{i}\|>`                                   |
        | `phi4mm`       | `<\|image_{i}\|>`                                   |
        | `qwen2vl`      | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` |
        | `qwen25vl`     | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` |
        | `gemma3`       | `<start_of_image>`                                  |

    === "Whisper"

        ```
        openarc add \
          --model-name <model-name> \
          --model-path <path/to/whisper> \
          --engine ovgenai \
          --model-type whisper \
          --device <target-device>
        ```

    === "Kokoro"

        ```
        openarc add \
          --model-name <model-name> \
          --model-path <path/to/kokoro> \
          --engine openvino \
          --model-type kokoro \
          --device CPU
        ```

    === "Qwen3-TTS"

        Qwen3-TTS has three modes, each selected by `--model-type` at add time. Inference parameters (speaker, voice description, reference audio, sampling settings) are supplied per-request via the API, not here.

        CPU and GPU device are supported.

        When GPU is selected as device, part of the model still runs on CPU. 

        Supported languages: `english`, `chinese`, `japanese`, `korean`, `german`, `french`, `spanish`, `italian`, `portuguese`, `russian`, `beijing_dialect`, `sichuan_dialect`. Pass `None` to auto-detect. See `demos/qwen3_tts_example.py` for a full request example.

        === "Custom voice"

            Pick a predefined speaker at inference time (`serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`):

            ```
            openarc add \
              --model-name <model-name> \
              --model-path <path/to/qwen3-tts> \
              --engine openvino \
              --model-type qwen3_tts_custom_voice \
              --device CPU
            ```

            ```python
            import os
            from openai import OpenAI
            from pathlib import Path

            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key=os.environ["OPENARC_API_KEY"],
            )

            response = client.audio.speech.create(
                model="<model-name>",
                input="Hello, this is a test.",
                extra_body={
                    "openarc_tts": {
                        "qwen3_tts": {
                            # --- content ---
                            "input": "Hello, this is a test.",
                            "speaker": "uncle_fu",       # serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan
                            "instruct": None,            # optional style instruction e.g. "Speak slowly and clearly."
                            "language": "english",       # None to auto-detect
                            # --- sampling ---
                            "max_new_tokens": 2048,
                            "do_sample": True,
                            "top_k": 50,
                            "top_p": 1.0,
                            "temperature": 0.9,
                            "repetition_penalty": 1.05,
                            "non_streaming_mode": True,
                            "subtalker_do_sample": True,
                            "subtalker_top_k": 50,
                            "subtalker_top_p": 1.0,
                            "subtalker_temperature": 0.9,
                            # --- streaming ---
                            "stream": True,
                            "stream_chunk_frames": 50,
                            "stream_left_context": 25,
                        }
                    }
                },
            )

            Path("speech.wav").write_bytes(response.content)
            ```

        === "Voice design"

            Describe the voice in free-form text at inference time:

            ```
            openarc add \
              --model-name <model-name> \
              --model-path <path/to/qwen3-tts> \
              --engine openvino \
              --model-type qwen3_tts_voice_design \
              --device CPU
            ```

            ```python
            import os
            from openai import OpenAI
            from pathlib import Path

            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key=os.environ["OPENARC_API_KEY"],
            )

            response = client.audio.speech.create(
                model="<model-name>",
                input="Hello, this is a test.",
                voice="alloy",
                extra_body={
                    "openarc_tts": {
                        "qwen3_tts": {
                            # --- content ---
                            "input": "Hello, this is a test.",
                            "voice_description": "A calm, deep male voice with a slight British accent.",
                            "language": "english",       # None to auto-detect
                            # --- sampling ---
                            "max_new_tokens": 2048,
                            "do_sample": True,
                            "top_k": 50,
                            "top_p": 1.0,
                            "temperature": 0.9,
                            "repetition_penalty": 1.05,
                            "subtalker_do_sample": True,
                            "subtalker_top_k": 50,
                            "subtalker_top_p": 1.0,
                            "subtalker_temperature": 0.9,
                            # --- streaming ---
                            "stream": True,
                            "stream_chunk_frames": 300,
                            "stream_left_context": 25,
                        }
                    }
                },
            )

            Path("speech.wav").write_bytes(response.content)
            ```

        === "Voice clone"

            Provide a reference WAV at inference time to clone a speaker:

            ```
            openarc add \
              --model-name <model-name> \
              --model-path <path/to/qwen3-tts> \
              --engine openvino \
              --model-type qwen3_tts_voice_clone \
              --device CPU
            ```

            ```python
            import base64
            import os
            from openai import OpenAI
            from pathlib import Path

            client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key=os.environ["OPENARC_API_KEY"],
            )

            ref_audio_b64 = base64.b64encode(Path("reference.wav").read_bytes()).decode()

            response = client.audio.speech.create(
                model="<model-name>",
                input="Hello, this is a test.",
                voice="alloy",
                extra_body={
                    "openarc_tts": {
                        "qwen3_tts": {
                            # --- content ---
                            "ref_audio_b64": ref_audio_b64,
                            "ref_text": "Transcript of the reference audio.",  # optional, enables ICL
                            "x_vector_only": False,      # True = x-vector only, skips ICL even if ref_text is set
                            "instruct": None,            # optional style instruction
                            "language": "english",       # None to auto-detect
                            # --- sampling ---
                            "max_new_tokens": 2048,
                            "do_sample": True,
                            "top_k": 50,
                            "top_p": 1.0,
                            "temperature": 0.9,
                            "repetition_penalty": 1.05,
                            "subtalker_do_sample": True,
                            "subtalker_top_k": 50,
                            "subtalker_top_p": 1.0,
                            "subtalker_temperature": 0.9,
                            # --- streaming ---
                            "stream": True,
                            "stream_chunk_frames": 300,
                            "stream_left_context": 25,
                        }
                    }
                },
            )

            Path("speech.wav").write_bytes(response.content)
            ```

    === "Qwen3-ASR"

        Qwen3-ASR long-form transcription — supports Qwen3-ASR-0.6B. Audio is chunked automatically at silence boundaries up to `max_chunk_sec` (default `30s`). This is not a hard limit and happens dynamically based

        ```
        openarc add \
          --model-name <model-name> \
          --model-path <path/to/qwen3-asr> \
          --engine openvino \
          --model-type qwen3_asr \
          --device CPU
        ```

        Use the `/v1/audio/transcriptions` endpoint with `openarc_asr` in the request body:
        I have not tested our implementation with any community tooling yet; however, all tests using the openai python library are passing, and usually that's enough. 

        For the options in `extra_body`, they will likely not have support in any third party tool you don't build from scratch. I'm working on improving how these can be configured. Currently, the behavior is modified per request, so you can tinker with performance on CPU and GPU. At this time NPU device is unsupported.

        ```python
        import json
        import os
        from pathlib import Path
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.environ["OPENARC_API_KEY"],
        )

        with Path("audio.wav").open("rb") as f:
            response = client.audio.transcriptions.create(
                model="<model-name>",
                file=f,
                response_format="verbose_json",
                extra_body={
                    "openarc_asr": json.dumps({
                        "qwen3_asr": {
                            "language": None,         # auto-detect, or e.g. "english"
                            "max_tokens": 1024,       # max tokens per chunk
                            "max_chunk_sec": 30.0,    # max audio chunk length in seconds
                            "search_expand_sec": 5.0, # silence-search window expansion
                            "min_window_ms": 100.0,   # minimum silence window in ms
                        }
                    })
                },
            )

        print(response.text)
        ```

    === "Advanced"

        `runtime-config` accepts many options to modify `openvino` runtime behavior for different inference scenarios. OpenArc reports C++ errors to the server when these fail, making experimentation easy.

        See OpenVINO documentation on [Inference Optimization](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference.html) to learn more about what can be customized.

        Not all options are designed for transformers, so `runtime-config` was implemented in a way where you get immediate feedback from the OpenVINO runtime after loading a model. Add an argument, load that model, get feedback from the server, run `openarc bench`. This makes iterating faster in an area where the documentation is sparse. The options listed here have been validated.

        Review the [pipeline-parallelism preview](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview) to learn how you can customize multi-device inference using the HETERO device plugin.

        === "Multi-GPU Pipeline Parallel"

            ```
            openarc add \
              --model-name <model-name> \
              --model-path <path/to/model> \
              --engine ovgenai \
              --model-type llm \
              --device HETERO:GPU.0,GPU.1 \
              --runtime-config "{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}"
            ```

        === "Tensor Parallel"

            Requires more than one CPU socket in a single node.

            ```
            openarc add \
              --model-name <model-name> \
              --model-path <path/to/model> \
              --engine ovgenai \
              --model-type llm \
              --device CPU \
              --runtime-config "{"MODEL_DISTRIBUTION_POLICY": "TENSOR_PARALLEL"}"
            ```

        === "Hybrid / CPU Offload"

            ```
            openarc add \
              --model-name <model-name> \
              --model-path <path/to/model> \
              --engine ovgenai \
              --model-type llm \
              --device HETERO:GPU.0,CPU \
              --runtime-config "{"MODEL_DISTRIBUTION_POLICY": "PIPELINE_PARALLEL"}"
            ```

        === "Speculative Decoding"

            ```
            openarc add \
              --model-name <model-name> \
              --model-path <path/to/model> \
              --engine ovgenai \
              --model-type llm \
              --device GPU.0 \
              --draft-model-path <path/to/draftmodel> \
              --draft-device CPU \
              --num-assistant-tokens 5 \
              --assistant-confidence-threshold 0.5
            ```

=== "list"

    Reads added configurations from `openarc_config.json`.

    Display all added models:
    ```
    openarc list
    ```

    Display config metadata for a specific model:
    ```
    openarc list \
      <model-name> \
      -v
    ```

    Remove a configuration:
    ```
    openarc list \
      --remove <model-name>
    ```

=== "serve"

    Starts the server.

    ```
    openarc serve start # defaults to 0.0.0.0:8000
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

=== "load"

    After using `openarc add` you can use `openarc load` to read the added configuration and load models onto the OpenArc server.

    OpenArc uses arguments from `openarc add` as metadata to make routing decisions internally; you are querying for correct inference code.

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
