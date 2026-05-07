
Tokenization strategy notes for OpenVINO GenAI ContinuousBatchingPipeline.

Key behavior to keep in mind for text-only LLMs:

1) Chat templating source precedence (later entries override earlier ones):
   - tokenizer_config.json["chat_template"]
   - processor_config.json["chat_template"]
   - chat_template.json["chat_template"]
   - openvino.Model rt_info["chat_template"] (embedded in tokenizer IR metadata)
   - If unsupported by GenAI, a simplified supported template is substituted.

2) In OpenVINO GenAI, the resolved chat template can come from model/tokenizer
   metadata (including tokenizer IR metadata), rather than directly loading
   chat_template.jinja at runtime.

3) ContinuousBatchingPipeline tokenization path:
   - Passing prompt: str -> template is applied, then prompt text is tokenized
     internally by openvino_genai.Tokenizer.
   - Passing input_ids: ov.Tensor -> tokenization is bypassed entirely.

4) Consequence for experiments:
   - Prompt-side token counts and shape are template-dependent.
   - input_ids-side benchmarks isolate generation/scheduling by removing runtime
     prompt tokenization variance.

References:
 - openvino_genai/py_openvino_genai.pyi:
   - ContinuousBatchingPipeline overloads (`prompt` vs `input_ids`)
   - Tokenizer template precedence note

---

