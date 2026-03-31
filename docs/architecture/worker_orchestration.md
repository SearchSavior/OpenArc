# Worker Orchestration Documentation

This document describes the worker system architecture, request routing, and how inference requests are processed.

## Architecture

```
Request → WorkerRegistry → Model Queue → Queue Worker → InferWorker → Model Instance
```

## WorkerPacket

Dataclass representing an inference request packet flowing through the system.


## Error Handling

### Inference Failures

If inference fails (exception caught in InferWorker):
1. Error stored in `packet.response` as `"Error: ..."`
2. Metrics set to None
3. QueueWorker detects error response
4. Triggers model unload via `registry.register_unload()`
5. Worker loop exits
6. Server remains unblocked and no workers stall

## Thread Safety

- Queues are thread-safe (`asyncio.Queue`)
- WorkerRegistry uses `asyncio.Lock` for queue/task dictionary access
- Each model has its own queue and worker, ensuring isolation

## Concurrency Model

- **Per-Model Workers**: Each loaded model has its own dedicated worker
- **Async Queues**: Requests are queued and processed asynchronously
- **Parallel Processing**: Multiple models can process requests concurrently
- **Streaming Support**: Streaming uses separate queue mechanism

## Design Patterns

### Queue-Based Processing
- Decouples request submission from execution
- Enables backpressure handling
- Supports multiple concurrent requests per model

### Worker Pattern
- Dedicated worker per model
- Long-running async loops
- Clean shutdown via None sentinel

### Future-Based Communication
- Non-streaming uses `asyncio.Future` for result communication
- Enables async/await pattern

### Queue-Based Streaming
- Streaming uses `asyncio.Queue` for token delivery
- Enables async iteration pattern
