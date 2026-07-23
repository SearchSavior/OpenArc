runtime_config


runtime_config is an OpenArc entrypoint to the *properties* way of configuring openvino runtime. These settings allow users to tune the behavior of openivno runtime without needing to change application logic and are meant to be "portable", requring no code changes. Since OpenArc 

OpenArc does not validate these, and OpenVINO upstream does not provide a way to check the behvaior of these settings in all cases. They can help you access hardware features not available to all devices like `SCHEDULING_CORE_TYPE` for more recent Intel CPUs, debug numeircal precision issues with `INFERENCE_PRECISION_HINT` or control `KV_CACHE_PRECISION`.

*properties* have the worst documentation in all of OpenVINO ecosystem, yet they are used everywhere in the openvino_notebooks, PRs and sometimes are even hardcoded depending on the needs of OpenVINO team. In that way, poking at these settings can drastically change performance but have less knobs than users of projects like `llama.cpp`, `vllm`, `sglang` are familiar with making. 


Even though we can learn from the source code what these settings do knowing when they are useful comes with practice  


ATTENTION_BACKEND     "SDPA", "PA"
KV_CACHE_PRECISION   "u4", "u8", "f16", "f32"
PERFORMANCE_HINT     "LATNENCY", "THROUGHPUT"
EXECUTION_MODE_HINT   "ACCURACY", "PERFORMANCE"
INFERENCE_PRECISION_HINT "f16", "f32"
MODEL_DISTRIBUTION_POLICY      "TENSOR_PARALLEL", "PIPELINE_PARALLEL"
ACTIVATIONS_SCALING_FACTOR: 
DYNAMIC_QUANTIZATION_GROUP_SIZE:  integer
ENABLE_HYPER_THREADING:        bool, defaults to true
SCHEDULING_CORE_TYPE:   "ANY_CORE", "ECORE_ONLY", "PCORE_ONLY"
LOG_LEVEL:      "ERR", "WARN", "INFO", "DEBUG", "TRACE" # might require building openvino