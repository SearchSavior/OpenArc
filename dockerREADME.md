# OpenARC Docker v2 - Build Package

## What's Included

- **Dockerfile.v2** - Clean production build
- **docker-compose.v2.yml** - Deployment configuration
- **README.md** - This file

## Key Features

✅ Pulls latest OpenARC from main (no version pinning - always gets latest)
✅ Race condition fix for model autoload (30-second health check loop)
✅ Intel GPU environment variables (multi-GPU support)
✅ Healthcheck with API key authentication
✅ Persistent config via volume mount
✅ CPU-only PyTorch (no CUDA bloat)
✅ OpenVINO GenAI nightly (latest Arc GPU support)

## Quick Start

### 1. Build Image

```bash
docker build -f Dockerfile.v2 -t openarc:latest .
```

### 2. Configure Environment

```bash
export MODEL_PATH=/path/to/your/openvino/models
export OPENARC_API_KEY=your-secret-key
```

### 3. Deploy

```bash
docker-compose -f docker-compose.v2.yml up -d
```

### 4. Verify

```bash
docker logs -f openarc
```

## Usage

### Add Model

```bash
docker exec -it openarc openarc add \
  --model-name qwen-4b \
  --model-path /models/qwen3-4b-ov \
  --engine ovgenai \
  --model-type llm \
  --device GPU.0
```

### Load Model

```bash
docker exec -it openarc openarc load qwen-4b
```

### Test Inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "qwen-4b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Environment Variables

### Application

- `OPENARC_API_KEY` - API authentication key (default: openarc-default-key)
- `OPENARC_AUTOLOAD_MODEL` - Model to auto-load on startup (optional)
- `MODEL_PATH` - Host path to model directory

### Intel GPU Tuning

- `NEOReadDebugKeys=1` - Enable Intel GPU tuning variables
- `OverrideGpuAddressSpace=48` - 48-bit GPU addressing for large models
- `EnableImplicitScaling=1` - Multi-GPU workload distribution

## Persistent Config

Config is stored in Docker volume `openarc-config` and mapped to `/persist/openarc_config.json`.

This survives container rebuilds and version upgrades.

## Troubleshooting

### Container shows unhealthy
- Check API key matches between env and healthcheck
- Verify server started: `docker logs openarc | grep "Uvicorn running"`

### Model autoload fails
- Check logs: `docker logs openarc`
- Verify model path is correct in container: `docker exec openarc ls /models`

### GPU not detected
- Verify host has Intel GPU drivers installed
- Check `/dev/dri` exists on host
- Run device detect: `docker exec openarc openarc tool device-detect`
