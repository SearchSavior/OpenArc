"""
Real end-to-end CB test: loads ContinuousBatchingPipeline from a local
OpenVINO model and drives it through CBInferDaemon / CBRouter.

Skips automatically unless openvino_genai is importable and the model dir
exists (set CB_TEST_MODEL or use the default download path).
"""
import asyncio
import importlib.util
import os
import unittest

MODEL_DIR = os.environ.get("CB_TEST_MODEL", "/tmp/lfm25")
HAS_GENAI = importlib.util.find_spec("openvino_genai") is not None
RUNNABLE = HAS_GENAI and os.path.isdir(MODEL_DIR)


@unittest.skipUnless(RUNNABLE, f"openvino_genai + model at {MODEL_DIR} required")
class CBEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.engine.ov_genai.continuous_batching.cb_adapter_llm import ArcCBLLM
        from src.server.models.registration import (
            EngineType,
            ModelLoadConfig,
            ModelType,
        )

        cls.loader = ModelLoadConfig(
            model_path=MODEL_DIR,
            model_name="cbm",
            model_type=ModelType.CB_LLM,
            engine=EngineType.OV_GENAI,
            device="CPU",
            runtime_config={
                "cache_size": 1,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 512,
            },
        )
        cls.arc = ArcCBLLM(cls.loader)
        try:
            cls.arc.load_model(cls.loader)
        except RuntimeError as exc:
            if "undeclared parameters" in str(exc):
                raise unittest.SkipTest(
                    "model IR / openvino runtime mismatch: "
                    f"{str(exc).splitlines()[-1]}"
                )
            raise

    def _gen_cfg(self, **kw):
        from src.server.models.ov_genai import OVGenAI_GenConfig

        base = dict(
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=24,
            stream=True,
            stream_chunk_tokens=1,
        )
        base.update(kw)
        return OVGenAI_GenConfig(**base)

    def test_infer_daemon_streams_text_metrics_none(self):
        async def run():
            from src.server.cb_daemons.cb_worker import CBInferDaemon, CBRequest

            d = CBInferDaemon("cbm", self.arc)
            d.start()
            req = CBRequest(gen_config=self._gen_cfg(), stream_queue=asyncio.Queue())
            req.int_id = 1
            await d.submit(req)

            chunks, metrics, saw_none = [], None, False
            while True:
                item = await asyncio.wait_for(req.stream_queue.get(), timeout=120)
                if item is None:
                    saw_none = True
                    break
                if isinstance(item, dict):
                    metrics = item["metrics"]
                else:
                    chunks.append(item)
            await d.stop()

            text = "".join(chunks)
            self.assertTrue(text.strip(), f"expected non-empty text, got {text!r}")
            self.assertTrue(saw_none)
            self.assertIsNotNone(metrics)
            self.assertGreater(metrics["input_token"], 0)
            self.assertGreater(metrics["new_token"], 0)
            self.assertEqual(
                metrics["total_token"],
                metrics["input_token"] + metrics["new_token"],
            )

        asyncio.run(run())

    def test_two_requests_interleave(self):
        async def run():
            from src.server.cb_daemons.cb_worker import CBInferDaemon, CBRequest

            d = CBInferDaemon("cbm", self.arc)
            d.start()
            reqs = []
            for i in (1, 2):
                r = CBRequest(gen_config=self._gen_cfg(), stream_queue=asyncio.Queue())
                r.int_id = i
                await d.submit(r)
                reqs.append(r)

            async def drain(r):
                out = []
                while True:
                    it = await asyncio.wait_for(r.stream_queue.get(), timeout=120)
                    if it is None:
                        return out
                    if not isinstance(it, dict):
                        out.append(it)

            t1, t2 = await asyncio.gather(drain(reqs[0]), drain(reqs[1]))
            await d.stop()
            self.assertTrue("".join(t1).strip())
            self.assertTrue("".join(t2).strip())

        asyncio.run(run())

    def test_chunk_tokens_gt_one(self):
        async def run():
            from src.server.cb_daemons.cb_worker import CBInferDaemon, CBRequest

            d = CBInferDaemon("cbm", self.arc)
            d.start()
            req = CBRequest(
                gen_config=self._gen_cfg(stream_chunk_tokens=4),
                stream_queue=asyncio.Queue(),
            )
            req.int_id = 1
            await d.submit(req)

            chunks = []
            while True:
                item = await asyncio.wait_for(req.stream_queue.get(), timeout=120)
                if item is None:
                    break
                if not isinstance(item, dict):
                    chunks.append(item)
            await d.stop()
            self.assertTrue("".join(chunks).strip())

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
