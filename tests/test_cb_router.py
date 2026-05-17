"""
Unit tests for CBRequestDaemon id assignment/forwarding and CBRouter
dispatch + load/unload bookkeeping. No openvino_genai required.
"""
import asyncio
import unittest

from src.server.cb_daemons.cb_router import CBRequestDaemon, CBRouter
from src.server.cb_daemons.cb_worker import CBRequest


class _FakeRegistry:
    """Captures subscribed callbacks; mimics ModelRegistry's subscriber API."""

    def __init__(self):
        self._on_loaded = []
        self._on_unloaded = []
        self._models = {}
        self._lock = asyncio.Lock()

    def add_on_loaded(self, cb):
        self._on_loaded.append(cb)

    def add_on_unloaded(self, cb):
        self._on_unloaded.append(cb)


class _Rec:
    def __init__(self, name, mtype, instance):
        self.model_name = name
        self.model_type = mtype
        self.model_instance = instance


class _FakeInfer:
    def __init__(self):
        self.submitted = []

    async def submit(self, req):
        self.submitted.append(req)


class RequestDaemonTests(unittest.TestCase):
    def test_assigns_incrementing_ids_and_forwards(self):
        async def run():
            infer = _FakeInfer()
            rd = CBRequestDaemon("m", infer)
            rd.start()
            r1 = CBRequest(gen_config=object(), stream_queue=asyncio.Queue())
            r2 = CBRequest(gen_config=object(), stream_queue=asyncio.Queue())
            await rd.submit(r1)
            await rd.submit(r2)
            await asyncio.sleep(0.05)
            await rd.stop()

            self.assertEqual([r.int_id for r in infer.submitted], [1, 2])

        asyncio.run(run())


class RouterTests(unittest.TestCase):
    def test_subscribes_to_registry(self):
        reg = _FakeRegistry()
        CBRouter(reg)
        self.assertEqual(len(reg._on_loaded), 1)
        self.assertEqual(len(reg._on_unloaded), 1)

    def test_stream_generate_unknown_model_raises(self):
        async def run():
            router = CBRouter(_FakeRegistry())
            with self.assertRaises(ValueError):
                async for _ in router.stream_generate("nope", object()):
                    pass

        asyncio.run(run())

    def test_non_cb_model_ignored(self):
        async def run():
            reg = _FakeRegistry()
            router = CBRouter(reg)
            await router._on_model_loaded(_Rec("x", "llm", object()))
            self.assertFalse(router.is_cb_model("x"))

        asyncio.run(run())

    def test_cb_load_starts_daemons_and_unload_stops(self):
        async def run():
            reg = _FakeRegistry()
            router = CBRouter(reg)

            class Adapter:
                def add_request(self, *a): ...
                def step(self): ...
                def has_non_finished_requests(self): return False
                def decode(self, ids): return ""
                def collect_metrics(self, **k): return {}

            rec = _Rec("cbm", "cb_llm", Adapter())
            await router._on_model_loaded(rec)
            self.assertTrue(router.is_cb_model("cbm"))

            await router._on_model_unloaded(rec)
            self.assertFalse(router.is_cb_model("cbm"))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
