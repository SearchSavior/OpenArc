import base64
import io

import librosa
import numpy as np
import openvino as ov
from openvino_genai import WhisperGenerationConfig, WhisperPipeline

model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Whisper/whisper-medium-int4-ov"
sample_audio_path = "/home/echo/Projects/OpenArc/src2/tests/john_steakly_armor_the_drop.wav"


@staticmethod
def audio_file_to_base64(audio_path: str) -> str:
    """
    Convert a local audio file to base64 string.
    """
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_string = base64.b64encode(audio_bytes).decode('utf-8')
        return base64_string


class OVGenAI_Whisper:
    def __init__(self, model_path: str, device: str = "GPU.0"):
        """
        Initialize an OpenVINO Whisper pipeline.
        """
        self.model_path = model_path
        self.pipeline = WhisperPipeline(model_path, device=device)

    def prepare_inputs(self, audio_base64: str) -> list[float]:
        """
        Prepare audio inputs from base64 string for the Whisper pipeline.
        """
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create a BytesIO object to simulate a file
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Load audio -> float32 mono at 16kHz
        audio, sr = librosa.load(audio_buffer, sr=16000, mono=True)
        
        # Convert to list[float] as required by WhisperPipeline
        audio_list = audio.astype(np.float32).tolist()
        
        return audio_list

    def transcribe(self, audio_base64: str) -> str:
        """
        Run transcription on a given base64 encoded audio.
        Language autodetection will be used (no language argument).
        """
        # Prepare audio inputs from base64
        audio_list = self.prepare_inputs(audio_base64)

        # Run transcription
        result = self.pipeline.generate(audio_list)

        # Access result as an object, not a dict
        return result.texts


if __name__ == "__main__":
    whisper = OVGenAI_Whisper(model_path)
    audio_base64 = audio_file_to_base64(sample_audio_path)
    transcript = whisper.transcribe(audio_base64)
    print("Transcription:")
    print(transcript)
