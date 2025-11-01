# Inference Engines Documentation


OpenArc supports three inference engines, each optimized for different model types:

- **OVGenAI**: OpenVINO GenAI pipeline (LLM, VLM, Whisper)
- **Optimum**: Optimum-Intel (Embedding, Reranker)
- **OpenVINO**: Native OpenVINO runtime (Kokoro TTS)

## Engine Architecture

```
src/engine/
├── ov_genai/
│   ├── llm.py           # OVGenAI_LLM
│   ├── vlm.py           # OVGenAI_VLM
│   ├── whisper.py        # OVGenAI_Whisper
│   ├── streamers.py      # ChunkStreamer
│   ├── continuous_batch_llm.py
│   └── continuous_batch_vlm.py
├── optimum/
│   ├── optimum_llm.py   # Optimum_LLM
│   ├── optimum_vlm.py   # Optimum_VLM
│   ├── optimum_emb.py   # Optimum_EMB
│   └── optimum_rr.py     # Optimum_RR
└── openvino/
    ├── kokoro.py         # OV_Kokoro
    └── kitten.py
```

## Class Hierarchy

### OVGenAI Engine

#### OVGenAI_LLM (`src/engine/ov_genai/llm.py`)

Text-to-text language model using OpenVINO GenAI LLMPipeline.

**Key Features:**
- Supports OpenAI-compatible chat message format with chat templates
- Tool calling support (tools parameter in messages)
- Streaming and non-streaming generation modes
- Multiple input formats: pre-encoded input_ids, raw prompts, and chat messages
- ChunkStreamer for batched token streaming (chunk_size > 1)
- Performance metrics collection (ttft, throughput, etc.)
- Uses AutoTokenizer for encoding, model tokenizer for decoding

#### OVGenAI_VLM (`src/engine/ov_genai/vlm.py`)

Vision-language model using OpenVINO GenAI VLMPipeline.

**Key Features:**
- Supports OpenAI-compatible multimodal message format with embedded images
- Tool calling support (tools parameter in messages)
- Streaming and non-streaming generation modes
- Extracts base64-encoded images from OpenAI message format
- Converts images to OpenVINO tensors for inference
- Inserts model-specific vision tokens at image positions
- Supports multiple images per request with proper token indexing
- ChunkStreamer for batched token streaming (chunk_size > 1)
- Performance metrics collection (ttft, throughput, etc.)
- Uses chat templates with vision token insertion

**Vision Token Types:**
- `internvl2`: `<image>`
- `llava15`: `<image>`
- `llavanext`: `<image>`
- `minicpmv26`: `(<image>./</image>)`
- `phi3vision`: `<|image_{i}|>`
- `phi4mm`: `<|image_{i}|>`
- `qwen2vl`: `<|vision_start|><|image_pad|><|vision_end|>`
- `qwen25vl`: `<|vision_start|><|image_pad|><|vision_end|>`
- `gemma3`: `<start_of_image>`

#### OVGenAI_Whisper (`src/engine/ov_genai/whisper.py`)

Automatic speech recognition using OpenVINO GenAI Whisper

**Key Features:**
- Processes base64-encoded audio
- Returns transcribed text and metrics
- Non-streaming only (Whisper processes entire audio)

#### ChunkStreamer (`src/engine/ov_genai/streamers.py`)

Custom streamer for chunked token streaming. Uses OpenVINO tokenizer, not AutoTokenizer for decode.

**Features:**
- Accumulates tokens into chunks
- Yields chunks when chunk_size reached
- Supports chunk_size > 1 for batched streaming

### Optimum Engine

#### Optimum_EMB (`src/engine/optimum/optimum_emb.py`)

Text-to-vector embedding model using Optimum-Intel.

**Key Features:**
- Uses `OVModelForFeatureExtraction`
- Implements last token pooling for embeddings
- Normalizes embeddings (L2 normalization)
- Supports flexible tokenizer configuration

**Token Pooling:**
- Handles left-padding vs right-padding
- Extracts last non-padding token embedding
- Normalizes to unit vectors

#### Optimum_RR (`src/engine/optimum/optimum_rr.py`)

Document reranking model using Optimum-Intel.

**Key Features:**
- Reranks documents based on query relevance
- Supports custom prefix/suffix/instruction
- Returns ranked document lists

### OpenVINO Engine

#### OV_Kokoro (`src/engine/openvino/kokoro.py`)

Text-to-speech model using native OpenVINO runtime.

**Key Features:**
- Processes text in chunks (character_count_chunk)
- Generates audio tensors per chunk
- Supports voice selection and language codes
- Speed control for speech generation
- Returns WAV audio format

**Voice Support:**
- Multiple languages (English, Japanese, Chinese, Spanish, etc.)
- Multiple voices per language
- Gender-specific voices

#