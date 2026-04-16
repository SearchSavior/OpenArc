import random
from transformers import AutoTokenizer

MODEL_PATH = "/mnt/Ironwolf-4TB/Models/Pytorch/Qwen3.5/Qwen3.5-9B/"
TARGET = 16384

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
vocab_size = tokenizer.vocab_size

for trial in range(10):
    raw_ids = random.sample(range(vocab_size), TARGET)

    decoded = tokenizer.decode(raw_ids, skip_special_tokens=False)
    stabilized_ids = tokenizer.encode(decoded, add_special_tokens=False)

    pre_truncate = len(stabilized_ids)
    stabilized_ids = stabilized_ids[:TARGET]

    calibrated_prompt = tokenizer.decode(stabilized_ids, skip_special_tokens=False)
    final_ids = tokenizer.encode(calibrated_prompt, add_special_tokens=False)

    drift = len(final_ids) - TARGET
    direction = "OVER" if pre_truncate >= TARGET else "SHORT"

    print(
        f"Trial {trial+1:2d}: "
        f"after_stabilize={pre_truncate:5d}  {direction:5s} by {abs(pre_truncate - TARGET):4d}  "
        f"after_truncate+rt={len(final_ids):5d}  drift={drift:+3d}"
    )
