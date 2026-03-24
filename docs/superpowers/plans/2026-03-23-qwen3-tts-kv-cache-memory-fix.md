# Qwen3-TTS KV Cache Memory Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix growing memory usage across TTS generation turns caused by OpenVINO KV cache retention.

**Architecture:** Three targeted changes to the engine layer — (1) switch stateful inference from `request.infer()` to `start_async(share_inputs=False)` + `wait()` so the OV runtime does not pin input numpy arrays, (2) retain compiled model references so infer requests can be recreated for full deallocation, (3) add post-generation `reset_state()` calls to release KV cache buffers between turns rather than holding them idle.

**Tech Stack:** OpenVINO Python API (`openvino`), NumPy

**Scope:** Two files only — `src/engine/openvino/qwen3_tts/qwen3_tts.py` and `src/engine/openvino/qwen3_tts/qwen3_tts_helpers.py`. No server/worker/routing changes.

---

## Context

The `OVQwen3TTS` engine uses OpenVINO stateful models for the talker and code predictor. These models maintain an internal KV cache that grows during autoregressive generation. Three issues cause memory to grow across turns:

1. **`request.infer(inputs)` defaults to `share_inputs=True`** — the OV runtime holds references to input numpy arrays (embeddings, RoPE slices) instead of copying them, preventing GC.
2. **Compiled model references (`talker_c`, `cp_c`) are local variables in `load_model()`** — they go out of scope, making it impossible to recreate infer requests for full buffer deallocation.
3. **No post-generation cleanup** — after `_run_loop()` returns, the talker's KV cache (prefill + all decode steps) sits allocated in memory until the *next* `generate()` call's `reset_state()`.

---

## File Map

| File | Changes |
|------|---------|
| `src/engine/openvino/qwen3_tts/qwen3_tts_helpers.py:228-234` | Modify `ov_stateful_infer()` to use `start_async` + `wait` with `share_inputs=False` |
| `src/engine/openvino/qwen3_tts/qwen3_tts.py:61-77` | Add `_talker_compiled` and `_cp_compiled` attributes to `__init__` |
| `src/engine/openvino/qwen3_tts/qwen3_tts.py:111-114` | Store compiled models before creating infer requests in `load_model()` |
| `src/engine/openvino/qwen3_tts/qwen3_tts.py:132-151` | Null out new compiled model refs in `unload_model()` |
| `src/engine/openvino/qwen3_tts/qwen3_tts.py:163-168` | Add post-generation cleanup in `_generate_sync()` |

---

### Task 1: Switch `ov_stateful_infer` to `start_async` with `share_inputs=False`

**Why:** `request.infer(inputs)` defaults to `share_inputs=True`, letting the OV runtime pin references to input numpy arrays. Switching to `start_async(share_inputs=False)` + `wait()` forces the runtime to copy input data to its own buffers, allowing the original numpy arrays to be garbage collected.

**Files:**
- Modify: `src/engine/openvino/qwen3_tts/qwen3_tts_helpers.py:228-234`

- [ ] **Step 1: Modify `ov_stateful_infer`**

Open `src/engine/openvino/qwen3_tts/qwen3_tts_helpers.py`. Replace the current implementation at lines 228-234:

```python
# CURRENT (lines 228-234):
@staticmethod
def ov_stateful_infer(request, inputs: dict) -> dict:
    request.infer(inputs)
    return {
        out.get_any_name(): request.get_tensor(out.get_any_name()).data.copy()
        for out in request.model_outputs
    }
```

With:

```python
@staticmethod
def ov_stateful_infer(request, inputs: dict) -> dict:
    request.start_async(inputs, share_inputs=False)
    request.wait()
    return {
        out.get_any_name(): request.get_tensor(out.get_any_name()).data.copy()
        for out in request.model_outputs
    }
```

The only change is replacing `request.infer(inputs)` with `request.start_async(inputs, share_inputs=False)` followed by `request.wait()`. The output extraction (`.data.copy()`) stays the same.

- [ ] **Step 2: Verify the server starts and a single TTS request works**

Run the server and send a test TTS request (any mode). Confirm audio is generated without errors. The output quality/content should be identical — this change only affects memory ownership of input buffers, not inference results.

- [ ] **Step 3: Commit**

```bash
git add src/engine/openvino/qwen3_tts/qwen3_tts_helpers.py
git commit -m "fix(tts): use share_inputs=False in stateful infer to prevent input pinning"
```

---

### Task 2: Retain compiled model references for infer request recreation

**Why:** Currently `talker_c` and `cp_c` are local variables in `load_model()` that go out of scope. Although the infer request keeps the compiled model alive internally, we cannot access the compiled model to call `create_infer_request()` again. Storing these references enables Task 3 to recreate fresh infer requests between turns, which is the only way to guarantee full KV cache buffer deallocation (since `reset_state()` may not shrink allocations).

**Files:**
- Modify: `src/engine/openvino/qwen3_tts/qwen3_tts.py:61-77` (`__init__`)
- Modify: `src/engine/openvino/qwen3_tts/qwen3_tts.py:111-114` (`load_model`)
- Modify: `src/engine/openvino/qwen3_tts/qwen3_tts.py:132-151` (`unload_model`)

- [ ] **Step 1: Add compiled model attributes to `__init__`**

In `src/engine/openvino/qwen3_tts/qwen3_tts.py`, in the `__init__` method, add two new attributes after line 69 (`self._cp_req = None`):

```python
# CURRENT (lines 68-69):
self._talker_req = None
self._cp_req = None

# CHANGE TO:
self._talker_req = None
self._cp_req = None
self._talker_compiled = None
self._cp_compiled = None
```

- [ ] **Step 2: Store compiled models in `load_model`**

In `load_model()`, change lines 111-114 from:

```python
# CURRENT (lines 111-114):
talker_c = core.compile_model(str(p / "talker.xml"), device)
self._talker_req = talker_c.create_infer_request()
cp_c = core.compile_model(str(p / "code_predictor.xml"), device)
self._cp_req = cp_c.create_infer_request()
```

To:

```python
self._talker_compiled = core.compile_model(str(p / "talker.xml"), device)
self._talker_req = self._talker_compiled.create_infer_request()
self._cp_compiled = core.compile_model(str(p / "code_predictor.xml"), device)
self._cp_req = self._cp_compiled.create_infer_request()
```

The only change is assigning to `self._talker_compiled` / `self._cp_compiled` instead of local variables `talker_c` / `cp_c`.

- [ ] **Step 3: Null out compiled model refs in `unload_model`**

In `unload_model()`, add two lines after the existing `self._cp_req = None` (line 140). The section should read:

```python
# CURRENT (lines 139-140):
self._talker_req = None
self._cp_req = None

# CHANGE TO:
self._talker_req = None
self._cp_req = None
self._talker_compiled = None
self._cp_compiled = None
```

- [ ] **Step 4: Verify the server starts and model load/unload works**

Start the server, load a qwen3_tts model, send a TTS request, unload the model. Confirm no errors.

- [ ] **Step 5: Commit**

```bash
git add src/engine/openvino/qwen3_tts/qwen3_tts.py
git commit -m "refactor(tts): retain compiled model refs for talker and code predictor"
```

---

### Task 3: Add post-generation KV cache cleanup

**Why:** After `_run_loop()` returns, the talker's internal KV cache holds the full state from the generation (prefill + all decoded frames). This memory sits idle until the next `generate()` call triggers `reset_state()`. By resetting state *immediately* after generation and recreating the infer requests, we release all KV cache buffers between turns.

**Files:**
- Modify: `src/engine/openvino/qwen3_tts/qwen3_tts.py:163-168` (`_generate_sync`)

- [ ] **Step 1: Add cleanup method**

Add a new private method to `OVQwen3TTS`, after `_generate_sync` (after line 168) and before `_generate_standard` (line 172). Insert:

```python
def _cleanup_kv_cache(self) -> None:
    """Release KV cache buffers between generation turns."""
    self._talker_req.reset_state()
    self._cp_req.reset_state()
    self._talker_req = self._talker_compiled.create_infer_request()
    self._cp_req = self._cp_compiled.create_infer_request()
```

This resets state (zeros out current buffers), then recreates fresh infer requests from the compiled models. The old requests — and their peak-sized internal buffers — become unreferenced and eligible for GC.

- [ ] **Step 2: Call cleanup after generation in `_generate_sync`**

Modify `_generate_sync` (lines 163-168) to wrap the generation call in a try/finally:

```python
# CURRENT (lines 163-168):
def _generate_sync(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
    if not self._loaded:
        raise RuntimeError("Call load_model() before generate()")
    if self.load_config.model_type == ModelType.QWEN3_TTS_VOICE_CLONE:
        return self._generate_voice_clone(gen_config)
    return self._generate_standard(gen_config)

# CHANGE TO:
def _generate_sync(self, gen_config: OV_Qwen3TTSGenConfig) -> tuple[np.ndarray, int]:
    if not self._loaded:
        raise RuntimeError("Call load_model() before generate()")
    try:
        if self.load_config.model_type == ModelType.QWEN3_TTS_VOICE_CLONE:
            return self._generate_voice_clone(gen_config)
        return self._generate_standard(gen_config)
    finally:
        self._cleanup_kv_cache()
```

The `finally` block ensures cleanup happens even if generation raises an exception.

- [ ] **Step 3: Verify multi-turn generation works**

Start the server and send at least 3 consecutive TTS requests. Confirm:
- All requests return valid audio
- No errors in server logs
- Memory does not grow monotonically (check with e.g. `ps -o rss -p <pid>` between requests)

- [ ] **Step 4: Commit**

```bash
git add src/engine/openvino/qwen3_tts/qwen3_tts.py
git commit -m "fix(tts): release KV cache buffers between generation turns"
```
