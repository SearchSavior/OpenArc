"""
Enum / class-mapping / scaffold tests for cb_llm and cb_vlm.
Requires pydantic (registration models); no openvino stack needed.
"""
import importlib.util
import unittest

from src.server.model_registry import MODEL_CLASS_REGISTRY
from src.server.models.registration import EngineType, ModelType

HAS_OV = importlib.util.find_spec("openvino") is not None


class RegistrationTests(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(ModelType.CB_LLM.value, "cb_llm")
        self.assertEqual(ModelType.CB_VLM.value, "cb_vlm")
        self.assertIs(ModelType("cb_llm"), ModelType.CB_LLM)
        self.assertIs(ModelType("cb_vlm"), ModelType.CB_VLM)

    def test_class_mapping(self):
        self.assertEqual(
            MODEL_CLASS_REGISTRY[(EngineType.OV_GENAI, ModelType.CB_LLM)],
            "src.engine.ov_genai.continuous_batching.cb_adapter_llm.ArcCBLLM",
        )
        self.assertEqual(
            MODEL_CLASS_REGISTRY[(EngineType.OV_GENAI, ModelType.CB_VLM)],
            "src.engine.ov_genai.continuous_batching.cb_adapter_vlm.ArcCBVLM",
        )

    @unittest.skipUnless(HAS_OV, "openvino not installed (src.engine eager import)")
    def test_cb_vlm_not_implemented(self):
        from src.engine.ov_genai.continuous_batching.cb_adapter_vlm import ArcCBVLM

        adapter = ArcCBVLM(load_config=None)
        with self.assertRaises(NotImplementedError):
            adapter.load_model(None)


if __name__ == "__main__":
    unittest.main()
