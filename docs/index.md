# OpenArc Documentation

Welcome to OpenArc documentation!  

This document collects information about the codebase structure, APIs, architecture and design patterns to help you explore the codebase.


- **[Server](./server.md)** - FastAPI server documentation with endpoint details
- **[Model Registration](./model_registration.md)** - How models are registered, loaded, and managed
- **[Worker Orchestration](./worker_orchestration.md)** - Worker system architecture and request routing
- **[Inference](./inference.md)** - Inference engines, class structure, and implementation details

### Architecture Overview

```
┌─────────────────┐
│   FastAPI       │  HTTP API Layer
│   Server        │  (OpenAI-compatible endpoints)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ WorkerRegistry  │  Request Routing & Orchestration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ModelRegistry   │  Model Lifecycle Management
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Inference      │  Engine-specific implementations
│  Engines        │  (OVGenAI, Optimum, OpenVINO)
└─────────────────┘
```

### Key Components

1. **Server** (`src/server/main.py`)
   - FastAPI application with OpenAI-compatible endpoints
   - Authentication middleware
   - Request/response handling

2. **Model Registry** (`src/server/model_registry.py`)
   - Model lifecycle management (load/unload)
   - Status tracking
   - Factory pattern for engine instantiation

3. **Worker Registry** (`src/server/worker_registry.py`)
   - Per-model worker queues
   - Request routing and orchestration
   - Async packet processing

4. **Inference Engines** (`src/engine/`)
   - **OVGenAI**: LLM, VLM, Whisper models
   - **Optimum**: Embedding, Reranker models
   - **OpenVINO**: Kokoro TTS models

## Supported Model Types

- **LLM**: Text-to-text language models
- **VLM**: Vision-language models (image-to-text)
- **Whisper**: Automatic speech recognition
- **Kokoro**: Text-to-speech
- **Embedding**: Text-to-vector embeddings
- **Reranker**: Document reranking

## Supported Libraries

- **OVGenAI**: OpenVINO GenAI pipeline (LLM, VLM, Whisper)
- **Optimum**: Optimum-Intel (Embedding, Reranker)
- **OpenVINO**: Native OpenVINO runtime (Kokoro TTS)

This project is about intel devices, so expect we may expand to other frameworks/libraries in the future.



