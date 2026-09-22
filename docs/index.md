---
icon: lucide/chess-rook
---


# Start Here


[![Discord](https://img.shields.io/discord/1341627368581628004?logo=Discord&logoColor=%23ffffff&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FmaMY7QjG)](https://discord.gg/Bzz9hax9Jq)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Echo9Zulu-yellow)](https://huggingface.co/Echo9Zulu)
[![Devices](https://img.shields.io/badge/Devices-CPU%2FGPU%2FNPU-blue)](https://github.com/openvinotoolkit/openvino)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SearchSavior/OpenArc)

Welcome to the OpenArc documentation! 

## Installation

- [Linux](install.md#linux)
- [Windows](install.md#windows)
- [Docker](install.md#docker)

## Commands

OpenArc includes a command line tool for controlling the server.

- [openarc list](commands.md#list) — List models added to the config.
- [openarc serve](commands.md#serve) — Start the OpenArc server.
- [openarc load](commands.md#load) — Load a model from the config.
- [openarc status](commands.md#status) — Check loaded models.
- [openarc bench](commands.md#bench) — Benchmarking tool for LLMs.
- [openarc tool](commands.md#tool) — OpenVINO utilities.

## Configuration

OpenArc uses a YAML based config system. See [Examples](configure.md#examples) for a config block for each model type.

- [LLM](configure.md#llm)
- [VLM](configure.md#vlm)
- [Kokoro](configure.md#kokoro)
- [Qwen3-ASR](configure.md#qwen3-asr)
- [Qwen3-TTS](configure.md#qwen3-tts)

## Performance

Scheduler, runtime properties, and multi-device recipes.

- [scheduler_config](performance.md#scheduler_config)
- [runtime_config](performance.md#runtime_config)
- [Multi-Device Inference](performance.md#multi-device-inference)
- [Speculative Decoding](performance.md#speculative-decoding)
- [Model Caching](performance.md#model-caching)

## API Usage

Request-time parameters passed via `extra_body` on the OpenAI-compatible API.

- [Qwen3-TTS](api_usage.md#qwen3-tts)
- [Qwen3-ASR](api_usage.md#qwen3-asr)


## Concepts

- [Tool and Reasoning Parsing](tool_use.md#tool-and-reasoning-parsing)
- [Out-of-Process Inference Workers](worker-processes.md) — VLM/LLM/Whisper run in supervised worker subprocesses so unload/load always get a clean OpenVINO Core


## Models

Models to get you started and where to find more!

OpenArc is deeply integrated with the Huggingface Ecosytem and has been written from the ground up to handle a ton of deployment complexity but still demands some calories to choose what models to use. 

We are working on improving this process with experimental GGUF support coming, as well as a new frontend application similar to LM-Studio!

Below are some models to get started which are known to work. My huggingface has many 

- [Model Sources](models.md#sources)
- [LLMs](models.md#llms)
- [VLMs](models.md#vlms)
- [Text to Speech](models.md#text-to-speech)
    - [Whisper](models.md#whisper)
    - [Qwen3-ASR](models.md#qwen3-asr)
- [Speech to Text](models.md#speech-to-text)
    - [Kokoro](models.md#kokoro)
    - [Qwen3-TTS](models.md#qwen3-tts)
- [Embedding](models.md#embedding)
- [Rerank](models.md#rerank)
