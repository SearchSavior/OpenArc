## Model Conversion

OpenVINO is an inference engine for leveraging diverse types of compute. To squeeze as much performance as possible from any hardware requires a bit more work than using the naive approach, especially once you have a usecase in mind and know what hardware you are using.

### The Naive Approach

OpenVINO defaults to **int8_asym** when setting "export=True" in both **OVModelForCausalLM.from_pretrained()** and the Optimum CLI Export Tool if no arguments for weight_format are passed. 

OpenArc has been designed for usecases which wander toward the bleeding edge of AI where users are expected to understand the nuance of datatypes, quantization strategies, calibration datasets, how these parameters contribute to accuracy/quality degredation across tasks. Perhaps you have just come from IPEX or (as of 2.5) 'vanilla' Pytorch and are looking to optimize a deployment.

For convience "export=False" is exposed on the /model/load endpoint; however I **strongly discourage** using it. To get the best performance from OpenVINO you have to get into the weeds.

### The Less Naive Approach to Model Conversion

Many Intel CPUs support INT8 but it isn't always the best choice. 

OpenVINO notebooks prove out that INT4 weight only compression coupled with quantization strategies like AWQ + Scale Estimation achieve better performance across the Intel device ecosystem with negligable accuracy loss. Still, different model architectures offer different performance reguardless of the chosen datatype; in practice it can be hard to predict how a model will perform so understanding how these parameter's work is essential to maximizing throughput by testing different configurations on the same target model.


### Why Speed Matters

Nvidia GPUs are faster and have a better open source backbone than Intel. However, Intel devices are cheaper by comparison. Even so, I don't want speed for the sake of being fast. OpenArc has been tooled for Agentic usecases and synthetic data generation where low throughput can damage workflow execution. 

If I want to dump some problem into a RoundRobin style multi-turn chat I am not sitting there waiting for 



Note: If you are using cloud compute which uses Intel devices it should still work 