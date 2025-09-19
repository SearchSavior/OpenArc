import numpy as np
import librosa
import openvino as ov
from openvino_genai import WhisperPipeline, WhisperGenerationConfig


model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Whisper/whisper-medium-int4-ov"
sample_audio_path = "/home/echo/Projects/OpenArc/src2/tests/john_steakly_armor_the_drop.wav"


class OVGenAI_Whisper:
    def __init__(self, model_path: str, device: str = "GPU.0"):
        """
        Initialize an OpenVINO Whisper pipeline.
        """
        self.model_path = model_path
        self.pipeline = WhisperPipeline(model_path, device=device)

    def transcribe(self, audio_path: str) -> str:
        """
        Run transcription on a given audio file.
        Language autodetection will be used (no language argument).
        """
        # Load audio -> float32 mono at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Convert to list[float] as required by WhisperPipeline
        audio_list = audio.astype(np.float32).tolist()

        # Use default config (autodetect language, transcribe mode)
  
        # Run transcription
        result = self.pipeline.generate(audio_list)

        # Access result as an object, not a dict
        return result.texts


if __name__ == "__main__":
    whisper = OVGenAI_Whisper(model_path)
    transcript = whisper.transcribe(sample_audio_path)
    print("Transcription:")
    print(transcript)
