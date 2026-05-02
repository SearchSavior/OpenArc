from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor

model_id = "google/gemma-4-E4B-it"

processor = AutoProcessor.from_pretrained(model_id)

model = OVModelForVisualCausalLM.from_pretrained(
    model_id,
    export=True,
    attn_implementation="sdpa",
)

model.save_pretrained("gemma4-e4b-sdpa-ov")
processor.save_pretrained("gemma4-e4b-sdpa-ov")