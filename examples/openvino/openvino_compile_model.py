

import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
# Initialize OpenVINO Core
core = ov.Core()

# Enable caching by setting the cache directory
cache_dir = "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Hermes-4-70B-int4_asym-ov/model_cache"
core.set_property({props.cache_dir: cache_dir})

# Read and compile the model
model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Llama/Hermes-4-70B-int4_asym-ov/openvino_model.xml"
device_name = "HETERO:GPU.0,GPU.1,GPU.2"  # or "CPU", "NPU", etc.

# Method 1: Read then compile
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name=device_name, config={hints.model_distribution_policy: "PIPELINE_PARALLEL"})

# Method 2: Compile directly (faster - skips separate read step)
#compiled_model = core.compile_model(model=model_path, device_name=device_name)


