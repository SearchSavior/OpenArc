# openarc bench

Benchmark `llm` performance with pseudo-random input tokens.

This approach follows [llama-bench](https://github.com/ggml-org/llama.cpp/blob/683fa6ba/tools/llama-bench/llama-bench.cpp#L1922), providing a baseline for the community to assess inference performance between `llama.cpp` backends and `openvino`.

To support different `llm` tokenizers, we need to standardize how tokens are chosen for benchmark inference. When you set `--p` we select `512` pseudo-random tokens as input_ids from the set of all tokens in the vocabulary.

`--n` controls the maximum amount of tokens we allow the model to generate; this bypasses `eos` and sets a hard upper limit.

## Default values

```
openarc bench <model-name> --p <512> --n <128> --r <5>
```

Which gives:

![openarc bench](../assets/openarc_bench_sample.png)

`openarc bench` also records metrics in a sqlite database `openarc_bench.db` for easy analysis.
