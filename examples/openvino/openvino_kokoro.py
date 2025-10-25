

import json
import time
from pathlib import Path
import soundfile as sf

import torch
import openvino as ov
from kokoro.model import KModel
from kokoro.pipeline import KPipeline   


# =====================================================================
# OpenVINO-backed Kokoro model
# =====================================================================
class OVKModel(KModel):
    def __init__(self, model_dir: Path, device: str):
        super().__init__()

        self.model_dir = Path(model_dir)

        # Load config.json
        with (self.model_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)

        self.vocab = config["vocab"]
        self.context_length = config["plbert"]["max_position_embeddings"]

        # Compile OpenVINO model
        start = time.perf_counter()
        core = ov.Core()
        self.model = core.compile_model(self.model_dir / "openvino_model.xml", device)
        self.compile_time_s = time.perf_counter() - start


    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        outputs = self.model([input_ids, ref_s, torch.tensor(speed)])
        return torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1])


# =====================================================================
# Timing helper
# =====================================================================
def time_inference(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - start)

# =====================================================================
# Example usage
# =====================================================================
if __name__ == "__main__":
    model_path = Path("path/to/model")

    # Initialize model + pipeline
    ov_model = OVKModel(model_path, device="CPU")
    print(f"Compile time: {ov_model.compile_time_s * 1000:.2f} ms")
    pipeline = KPipeline(model=ov_model, lang_code="a")

    input_text = (
        """
        Kokoro doesn't do well with emotional language, so if you use it with an text model, 
        include something the LLM instructions to guide its language to be "emotionally neutral" 
        before piping to Kokoro. Actually, I have no idea if that's true, or even works but needed some text
        for this example.
        """
    )

    with torch.no_grad():
        generator = pipeline(input_text, voice="af_heart")
        result, elapsed_s = time_inference(next, generator)

    print(f"Generated audio with {len(result.audio)} samples at 24kHz")
    print(f"Inference time: {elapsed_s * 1000:.2f} ms")
    
    # Save as WAV file
    output_path = "kokoro_output.wav"
    sf.write(output_path, result.audio, 24000)  # 24kHz sample rate
    print(f"Audio saved to: {output_path}")
