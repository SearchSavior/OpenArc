## Setting up your first model

### Disclaimer

The following assumes that you have all software dependencies installed.

### Getting Started

1. Download your model of choice from hugging face (I use the `hf` CLI app for this). See [models](models.md) for details.
```bash
hf auth login
hf download Echo9Zulu/Qwen3-1.7B-int8_asym-ov --local-dir ~/models/huggingface/Echo9Zulu/Qwen3-1.7B-int8_asym-ov
```

2. Check your device
```bash
openarc tool device-detect
```
```output
Detecting OpenVINO devices...
┏━━━━┳━━━━━━━━┓
┃ I… ┃ Device ┃
┡━━━━╇━━━━━━━━┩
│ 1  │ CPU    │
│ 2  │ GPU    │
└────┴────────┘

 Sanity test passed: found 2 device(s)
```
(tip: check the output of `openarc tool device-props` to find the hardware support for quantisation `fp32`, `fp16` or `int8`)

3. add the model to openarc locally
```bash
openarc add --model-name Qwen3-1.7B-int8_asym-ovgen --model-path ~/models/huggingface/Echo9Zulu/Qwen3-1.7B-int8_asym-ov --engine ovgenai --model-type llm --device GPU
```
```output
Model configuration saved: Qwen3-1.7B-int8_asym-ovgen
Use 'openarc load Qwen3-1.7B-int8_asym-ovgen' to load this model.
```

4. start the server
```bash
openarc serve start
```
```output

Configuration saved to: /home/steinb95/software/openarc/repo/openarc_config.json
OPENARC_API_KEY_REQUIRED=False [Clients do not need to authenticate.]
Starting OpenArc server on 0.0.0.0:8000
2026-05-29 10:34:10,934 - INFO - Launching  0.0.0.0:8000
2026-05-29 10:34:10,934 - INFO - --------------------------------
2026-05-29 10:34:10,934 - INFO - OpenArc endpoints:
2026-05-29 10:34:10,934 - INFO -   - POST   /openarc/load           Load a model
2026-05-29 10:34:10,934 - INFO -   - POST   /openarc/unload         Unload a model
2026-05-29 10:34:10,934 - INFO -   - GET    /openarc/status         Get model status
2026-05-29 10:34:10,934 - INFO -   - GET    /openarc/metrics            Get hardware telemetry
2026-05-29 10:34:10,934 - INFO -   - POST   /openarc/models/update      Update model configuration
2026-05-29 10:34:10,934 - INFO -   - POST   /openarc/bench              Run inference benchmark
2026-05-29 10:34:10,934 - INFO -   - GET    /openarc/downloader         List active model downloads
2026-05-29 10:34:10,934 - INFO -   - POST   /openarc/downloader         Start a model download
2026-05-29 10:34:10,934 - INFO -   - DELETE /openarc/downloader         Cancel a model download
2026-05-29 10:34:10,934 - INFO -   - POST   /openarc/downloader/pause   Pause a model download
2026-05-29 10:34:10,934 - INFO -   - POST   /openarc/downloader/resume  Resume a model download
2026-05-29 10:34:10,934 - INFO - --------------------------------
2026-05-29 10:34:10,934 - INFO - OpenAI compatible endpoints:
2026-05-29 10:34:10,934 - INFO -   - GET    /v1/models
2026-05-29 10:34:10,934 - INFO -   - POST   /v1/chat/completions
2026-05-29 10:34:10,934 - INFO -   - POST   /v1/audio/transcriptions: Whisper only
2026-05-29 10:34:10,934 - INFO -   - POST   /v1/audio/speech: Kokoro only
2026-05-29 10:34:10,934 - INFO -   - POST   /v1/embeddings
2026-05-29 10:34:10,934 - INFO -   - POST   /v1/rerank
2026-05-29 10:34:15,143 - INFO - Started server process [24426]
2026-05-29 10:34:15,144 - INFO - Waiting for application startup.
2026-05-29 10:34:15,144 - INFO - Application startup complete.
2026-05-29 10:34:15,144 - INFO - Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

5. load the model (in another shell perhaps)
```bash
openarc load Qwen3-1.7B-int8_asym-ov
```
```output
loading Qwen3-1.7B-int8_asym-ovgen
...working
Qwen3-1.7B-int8_asym-ovgen loaded!

────────────────────────────────────────────────────────────
All models loaded! (1/1)
Use 'openarc status' to see loaded models.
```

6. Test and Enjoy!
```bash
$ curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-1.7B-int8_asym-ovgen",
    "messages": [
      {"role": "user", "content": "Who invented the Transformer architecture?"}
    ]
  }'
```
```output
{"id":"ov-8a9f2b3404e6490ab3742d1e","object":"chat.completion","created":1780044148,"model":"Qwen3-1.7B-int8_asym-ovgen","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nOkay, so I need to figure out who invented the Transformer architecture. 
// ...
The foundational paper, *\"Attention Is All You Need,\"* was co-authored by **James V. McCann**, **Ashish Mittal**, **Shruti Nair**, **Nikita Erkmen**, **Youssef Ghazi**, and **Alon Oseran**, among others. \n\nThe paper, published in **2017**, introduced the **Transformer model**, which revolutionized natural language processing (NLP) by employing **self-attention mechanisms** to enable parallel processing of input sequences, even without relying on recurrent neural networks (RNNs). This innovation allowed transformers to handle long-range dependencies more efficiently and scale to large datasets.\n\nKey contributors include:\n- **Ilya Sutskever** (a researcher at DeepMind and later a leading figure in AI).\n- **Ashish Mittal** (a key member of the DeepMind team).\n- **James V. McCann** (a former DeepMind researcher).\n- **Nikita Erkmen** and others from the same team.\n\nThe Transformer became a cornerstone of modern NLP, underpinning models like BERT, GPT, and others, and is widely used in tasks such as machine translation, text generation, and more."},"finish_reason":"stop"}],"usage":{"prompt_tokens":14,"completion_tokens":1150,"total_tokens":1164},"metrics":{"load_time (s)":3.36,"ttft (s)":0.05,"tpot (ms)":19.48179,"prefill_throughput (tokens/s)":265.72,"decode_throughput (tokens/s)":51.32999,"decode_duration (s)":22.43743,"input_token":14,"new_token":1150,"total_token":1164,"stream":false}}%
```
> **Be aware of hallucinations**: The paper "Attention is all you need" was authored by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, see [here](https://arxiv.org/abs/1706.03762).
