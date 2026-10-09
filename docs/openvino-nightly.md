# OpenVINO nightly builds

OpenArc pins stable `openvino-genai` (see `dependencies` in `pyproject.toml`).
Some models require a newer nightly build of OpenVINO / GenAI. Example:
`OpenVINO/Qwen3.8-27B-int4-ov` requires OpenVINO 2026.4+ and **segfaults while
loading** on the pinned 2026.3.1 stable.

## Enabling the nightly (managed by uv)

The `nightly-ov` dependency group in `pyproject.toml` carries minimum nightly
versions for `openvino`, `openvino-genai`, and `openvino-tokenizers`:

```bash
uv sync --group nightly-ov \
  --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly \
  --index-strategy unsafe-best-match
```

Note: a plain `uv sync` afterwards resolves back to the stable pins and will
revert the environment. Either always sync with `--group nightly-ov`, or use
`uv run --no-sync ...` after installing.

## Manual override (no lockfile changes)

```bash
uv pip install --pre -U openvino openvino-genai openvino-tokenizers \
  --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly \
  --index-strategy unsafe-best-match
uv run --no-sync ...   # plain `uv run` / `uv sync` reverts to the stable pins
```

## Known-good nightly

| Date | openvino | openvino-genai / openvino-tokenizers | Used for |
|---|---|---|---|
| 2026-10-07 | 2026.5.0.dev20261007 | 2026.5.0.0.dev20261007 | Qwen3.8-27B-int4-ov (VLM engine) |

Record the exact nightly in every benchmark result produced while it is
installed.

## Companion upgrades (2026-10-08)

- `transformers` was upgraded 5.0.0 → 5.19.0 (5.0.0 does not recognize the
  `qwen3_5` architecture; needed for any Qwen3.5/3.8 export work) and
  `optimum[openvino]` to the latest release. Note: even with these,
  optimum-intel cannot export qwen3_5 for `text-generation-with-past` yet
  (see PLAN.md 3.2), so Qwen3.5/3.8 IRs still come from pre-exported hubs.
- `chat_template_kwargs` does **not** reach the Qwen3.5 chat template through
  `apply_chat_template` on transformers 5.x — pass `enable_thinking` as a
  direct keyword instead (OpenArc's VLM path already splats it correctly).
