# OpenArc Server Documentation

This document describes the FastAPI server implementation, endpoints, and API structure.

## Table of Contents

- [Overview](#overview)
- [Server Architecture](#server-architecture)
  - [Key Components](#key-components)
- [Authentication](#authentication)
- [CORS Configuration](#cors-configuration)
- [Endpoints](#endpoints)
  - [OpenArc Internal Endpoints](#openarc-internal-endpoints)
    - [`POST /openarc/load`](#post-openarcload)
    - [`POST /openarc/unload`](#post-openarcunload)
    - [`GET /openarc/status`](#get-openarcstatus)
    - [`POST /openarc/bench`](#post-openarcbench)
  - [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
    - [`GET /v1/models`](#get-v1models)
    - [`POST /v1/chat/completions`](#post-v1chatcompletions)
    - [`POST /v1/completions`](#post-v1completions)
    - [`POST /v1/audio/transcriptions`](#post-v1audiotranscriptions)
    - [`POST /v1/audio/speech`](#post-v1audiospeech)
    - [`POST /v1/embeddings`](#post-v1embeddings)
    - [`POST /v1/rerank`](#post-v1rerank)
- [Request Models](#request-models)
  - [OpenAIChatCompletionRequest](#openaichatcompletionrequest)
  - [OpenAICompletionRequest](#openaicompletionrequest)
  - [OpenAIWhisperRequest](#openaiwhisperrequest)
  - [OpenAIKokoroRequest](#openaikokororequest)
  - [EmbeddingsRequest](#embeddingsrequest)
  - [RerankRequest](#rerankrequest)
- [Tool Calling Support](#tool-calling-support)
  - [Parser Implementation](#parser-implementation)
- [Metrics](#metrics)
- [Startup Models](#startup-models)

## Overview

The OpenArc server is built with FastAPI and provides OpenAI-compatible endpoints for inference. The server is located in `src/server/main.py`.

## Server Architecture



### Key Components

- **FastAPI Application**: Main application instance with lifespan events
- **Model Registry**: Manages model lifecycle (load/unload)
- **Worker Registry**: Routes requests to appropriate workers
- **Authentication**: Bearer token authentication via `OPENARC_API_KEY`

## Authentication

All endpoints require authentication via Bearer token:

```python
Authorization: Bearer <OPENARC_API_KEY>
```

The API key is configured via the `OPENARC_API_KEY` environment variable.

## Endpoints

### OpenArc Internal Endpoints

#### `POST /openarc/load`

Load a model onto the server.

**Request Body:**
```json
{
  "model_path": "/path/to/model",
  "model_name": "my-model",
  "model_type": "llm",
  "engine": "ovgenai",
  "device": "GPU.0",
  "runtime_config": {},
  "vlm_type": null
}
```

**Response:**
```json
{
  "model_id": "unique-model-id",
  "model_name": "my-model",
  "status": "loaded"
}
```

**Status Codes:**
- `200`: Model loaded successfully
- `400`: Invalid request (e.g., model name already exists)
- `500`: Loading failed

#### `POST /openarc/unload`

Unload a model from the server.

**Request Body:**
```json
{
  "model_name": "my-model"
}
```

**Response:**
```json
{
  "model_name": "my-model",
  "status": "unloading"
}
```

**Status Codes:**
- `200`: Unload initiated
- `404`: Model not found
- `500`: Unload failed

#### `GET /openarc/status`

Get status of all loaded models.

**Response:**
```json
{
  "total_loaded_models": 2,
  "models": [
    {
      "model_name": "my-model",
      "model_type": "llm",
      "engine": "ovgenai",
      "device": "GPU.0",
      "runtime_config": {},
      "status": "loaded",
      "time_loaded": "2024-01-01T00:00:00"
    }
  ],
  "openai_model_names": ["my-model"]
}
```

#### `POST /openarc/bench`

Benchmark model performance with pre-encoded input IDs.

**Request Body:**
```json
{
  "model": "my-model",
  "input_ids": [1, 2, 3, ...],
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": 50,
  "repetition_penalty": 1.0
}
```

**Response:**
```json
{
  "metrics": {
    "ttft": 0.123,
    "prefill_throughput": 100.5,
    "decode_throughput": 50.2,
    "decode_duration": 2.5,
    "tpot": 0.025,
    "input_token": 512,
    "new_token": 128,
    "total_token": 640
  }
}
```

### OpenAI-Compatible Endpoints

#### `GET /v1/models`

List all available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "my-model",
      "object": "model",
      "created": 1704067200,
      "owned_by": "OpenArc"
    }
  ]
}
```

#### `POST /v1/chat/completions`

Chat completions endpoint for LLM and VLM models.

**Request Body:**
```json
{
  "model": "my-model",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "tools": [],
  "stream": false,
  "temperature": 1.0,
  "max_tokens": 512,
  "top_p": 1.0,
  "top_k": 50,
  "repetition_penalty": 1.0
}
```

**Response (non-streaming):**
```json
{
  "id": "ov-abc123...",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "my-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  },
  "metrics": {
    "ttft": 0.123,
    "prefill_throughput": 100.5,
    "decode_throughput": 50.2
  }
}
```

**Streaming Response:**
Server-Sent Events (SSE) format:
```
data: {"id": "ov-abc123...", "object": "chat.completion.chunk", ...}
data: {"id": "ov-abc123...", "object": "chat.completion.chunk", ...}
data: [DONE]
```

#### `POST /v1/completions`

Text completions endpoint for LLM models (legacy endpoint).

**Request Body:**
```json
{
  "model": "my-model",
  "prompt": "The capital of France is",
  "stream": false,
  "temperature": 1.0,
  "max_tokens": 512
}
```

**Response:**
```json
{
  "id": "ov-abc123...",
  "object": "text_completion",
  "created": 1704067200,
  "model": "my-model",
  "choices": [
    {
      "index": 0,
      "text": " Paris.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 2,
    "total_tokens": 7
  }
}
```

#### `POST /v1/audio/transcriptions`

Transcribe audio using Whisper models.

**Request Body:**
```json
{
  "model": "whisper-model",
  "audio_base64": "base64-encoded-audio-data"
}
```

**Response:**
```json
{
  "text": "Transcribed text here",
  "metrics": {
    "input_token": 100,
    "new_token": 50,
    "total_token": 150
  }
}
```

#### `POST /v1/audio/speech`

Generate speech using Kokoro TTS models.

**Request Body:**
```json
{
  "model": "kokoro-model",
  "input": "Hello, world!",
  "voice": "af_heart",
  "speed": 1.0,
  "language": "a",
  "response_format": "wav"
}
```

**Response:**
Returns WAV audio file as binary stream with `Content-Type: audio/wav`.

#### `POST /v1/embeddings`

Generate text embeddings.

**Request Body:**
```json
{
  "model": "embedding-model",
  "input": "Text to embed",
  "dimensions": null,
  "encoding_format": "float",
  "config": {
    "max_length": 512,
    "padding": true,
    "truncation": true
  }
}
```

**Response:**
```json
{
  "id": "ov-abc123...",
  "object": "list",
  "created": 1704067200,
  "model": "embedding-model",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "embedding": [0.1, 0.2, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

#### `POST /v1/rerank`

Rerank documents based on a query.

**Request Body:**
```json
{
  "model": "reranker-model",
  "query": "search query",
  "documents": ["doc1", "doc2", "doc3"],
  "prefix": "<|im_start|>system\n...",
  "suffix": "<|im_end|>\n...",
  "instruction": "Given a search query..."
}
```

**Response:**
```json
{
  "id": "ov-abc123...",
  "object": "list",
  "created": 1704067200,
  "model": "reranker-model",
  "data": [
    {
      "index": 0,
      "object": "ranked_documents",
      "ranked_documents": ["doc2", "doc1", "doc3"]
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "total_tokens": 50
  }
}
```

## Request Models

### OpenAIChatCompletionRequest
- `model`: str
- `messages`: List[Dict]
- `tools`: Optional[List[Dict]]
- `stream`: Optional[bool]
- `temperature`: Optional[float]
- `max_tokens`: Optional[int]
- `stop`: Optional[List[str]]
- `top_p`: Optional[float]
- `top_k`: Optional[int]
- `repetition_penalty`: Optional[float]
- `do_sample`: Optional[bool]
- `num_return_sequences`: Optional[int]

### OpenAICompletionRequest
- `model`: str
- `prompt`: Union[str, List[str]]
- `stream`: Optional[bool]
- `temperature`: Optional[float]
- `max_tokens`: Optional[int]
- `stop`: Optional[List[str]]
- `top_p`: Optional[float]
- `top_k`: Optional[int]
- `repetition_penalty`: Optional[float]
- `do_sample`: Optional[bool]
- `num_return_sequences`: Optional[int]

### OpenAIWhisperRequest
- `model`: str
- `audio_base64`: str

### OpenAIKokoroRequest
- `model`: str
- `input`: str
- `voice`: Optional[str]
- `speed`: Optional[float]
- `language`: Optional[str]
- `response_format`: Optional[str]

### EmbeddingsRequest
- `model`: str
- `input`: Union[str, List[str], List[List[str]]]
- `dimensions`: Optional[int]
- `encoding_format`: Optional[str]
- `user`: Optional[str]
- `config`: Optional[PreTrainedTokenizerConfig]

### RerankRequest
- `model`: str
- `query`: str
- `documents`: List[str]
- `prefix`: Optional[str]
- `suffix`: Optional[str]
- `instruction`: Optional[str]

## Tool Calling Support

OpenArc supports OpenAI-compatible tool calling. Tools are parsed from model output using regex pattern matching for JSON objects containing `name` and `arguments` fields.

Tool calls are detected in streaming and non-streaming modes:
- **Streaming**: Tool calls are detected incrementally and streamed as structured chunks
- **Non-streaming**: Tool calls are parsed from the final output

### Parser Implementation

The `parse_tool_calls()` function searches for JSON objects in the model's text output and converts them to OpenAI-compatible tool call format.

**Input Format (Model Output):**

The parser expects JSON objects embedded in the text with the following structure:

```json
{
  "name": "function_name",
  "arguments": {
    "arg1": "value1",
    "arg2": "value2"
  }
}
```

**Input to the parser from a model:**

```
The user wants to know the weather. {"name": "get_weather", "arguments": {"location": "San Francisco", "units": "celsius"}} I'll check that for you.
```

**Output Format (OpenAI-Compatible):**

Parser returns a list of tool call objects in OpenAI format:

```json
[
  {
    "id": "call_abc123def456...",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"San Francisco\", \"units\": \"celsius\"}"
    }
  }
]
```

**Parser Behavior:**

- Searches for JSON objects using regex pattern: `\{(?:[^{}]|(?:\{[^{}]*\}))*\}`
- Validates that each JSON object contains both `name` and `arguments` fields
- Generates unique IDs in format `call_{24-char-hex}`
- Converts `arguments` to JSON string (required by OpenAI format)
- Returns `None` if no valid tool calls are found

**Example Response (Non-Streaming):**

When tool calls are detected, the response includes:

```json
{
  "id": "ov-abc123...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123def456...",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"San Francisco\", \"units\": \"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

**Example Response (Streaming):**

Tool calls are streamed as structured chunks:

```
data: {"id": "ov-abc123...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_abc123...", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}}]}
data: {"id": "ov-abc123...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"location\": \"San Francisco\"}"}}]}}]}
data: {"id": "ov-abc123...", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
data: [DONE]
```

## Metrics

All inference endpoints return performance metrics:
- `ttft`: Time to first token
- `prefill_throughput`: Prefill tokens per second
- `decode_throughput`: Decode tokens per second
- `decode_duration`: Total decode duration
- `tpot`: Time per output token
- `input_token`: Number of input tokens
- `new_token`: Number of generated tokens
- `total_token`: Total tokens processed

## Startup Models

Models can be automatically loaded on server startup via the `OPENARC_STARTUP_MODELS` environment variable:

```bash
export OPENARC_STARTUP_MODELS="model1,model2,model3"
```

The server will read `openarc_config.json` and load the specified models during startup.

