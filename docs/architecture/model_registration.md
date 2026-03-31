# Model Registration Documentation

This document describes the model registration system, lifecycle management, and architectural patterns.

## Overview

The Model Registry (`src/server/model_registry.py`) manages the lifecycle of all models in OpenArc using a registry pattern with async background loading and a factory pattern for engine instantiation. 

## Architecture Patterns

### Registry Pattern

The `ModelRegistry` maintains a central dictionary of all loaded models, tracking their status and lifecycle state. It is a volatile in memory datastore used internally.

**Key Components:**
- **ModelRecord**: Tracks model state (LOADING, LOADED, FAILED)
- **Async Lock**: Ensures thread-safe concurrent access
- **Event System**: Callbacks for lifecycle events

### Factory Pattern

Models are instantiated via a factory that maps `(engine, model_type)` tuples to concrete engine classes:

The factory dynamically imports and instantiates the appropriate class based on configuration.

### Event System

The registry fires events when models are loaded or unloaded, allowing other components (like `WorkerRegistry`) to react:

```python
# Subscribe to events
registry.add_on_loaded(on_model_loaded)
registry.add_on_unloaded(on_model_unloaded)
```

## Model Lifecycle

```
┌─────────────┐
│   REQUEST   │
│ LOAD MODEL  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   CREATE    │
│ MODEL RECORD│
│ (LOADING)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  SPAWN      │
│ LOAD TASK   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  FACTORY    │
│ INSTANTIATE │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   UPDATE    │
│  STATUS TO  │
│  LOADED     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   FIRE      │
│  CALLBACKS  │
└─────────────┘
```

## Key Classes

### ModelLoadConfig

Pydantic model defining model configuration.

### ModelRecord

Dataclass tracking a registered model's state, instance, and metadata. Distinguishes between private (internal) and public (API-exposed) fields.

### ModelRegistry

Central registry implementing:
- **Async Loading**: Background tasks for model loading/unloading
- **Status Tracking**: LOADING → LOADED → FAILED states
- **Factory Integration**: Delegates instantiation to factory
- **Event Notifications**: Fires callbacks on lifecycle changes

## Thread Safety

All registry operations are protected by `asyncio.Lock` for thread-safe concurrent access. The registry maintains separate private model IDs while exposing public model names for API access.

## Integration

The `WorkerRegistry` subscribes to model lifecycle events to automatically spawn workers when models load and clean up when they unload. 
