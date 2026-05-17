"""
Unit tests for the continuous-batching worker decode/finalize logic.

These exercise CBInferDaemon._emit / _finalize / _admit_request with fakes and
need no openvino_genai / openvino. The full step-loop (CBInferDaemon._run)
lazily imports openvino_genai; an integration test for it is included but
skipped automatically when the runtime stack is unavailable.
"""
import asyncio
import importlib.util
import unittest

from src.server.cb_daemons.cb_worker import CBInferDaemon, CBRequest, _ActiveState

HAS_GENAI = importlib.util.find_spec("openvino_genai") is not None


class _GenCfg:
    def __init__(self, stream_chunk_tokens=1):
        self.stream_chunk_tokens = stream_chunk_tokens


class FakeArc:
    """Decodes ids as 1 char per id from a fixed alphabet; cumulative-safe."""

    def __init__(self):
        self.metrics_calls = []

    def decode(self, ids):
        # Each id 0..25 -> 'a'..'z'; deterministic and prefix-stable.
        return "".join(chr(ord("a") + (i % 26)) for i in ids)

    def collect_metrics(self, input_token, new_token):
        self.metrics_calls.append((input_token, new_token))
        return {
            "input_token": input_token,
            "new_token": new_token,
            "total_token": input_token + new_token,
        }


def _state(arc, gen_cfg):
    req = CBRequest(gen_config=gen_cfg, stream_queue=asyncio.Queue())
    return _ActiveState(request=req, handle=object(), n_input_tokens=7), req


async def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


class EmitChunkTokens1(unittest.TestCase):
    def test_token_by_token_delta(self):
        async def run():
            arc = FakeArc()
            d = CBInferDaemon("m", arc)
            st, req = _state(arc, _GenCfg(stream_chunk_tokens=1))

            st.generated_ids.extend([0])          # "a"
            await d._emit(st, final=False)
            st.generated_ids.extend([1])          # "ab"
            await d._emit(st, final=False)
            st.generated_ids.extend([2])          # "abc"
            await d._emit(st, final=False)

            self.assertEqual(await _drain(req.stream_queue), ["a", "b", "c"])

        asyncio.run(run())


class EmitChunkTokensN(unittest.TestCase):
    def test_emit_only_on_boundary(self):
        async def run():
            arc = FakeArc()
            d = CBInferDaemon("m", arc)
            st, req = _state(arc, _GenCfg(stream_chunk_tokens=3))

            st.generated_ids.extend([0])
            await d._emit(st, final=False)        # 1 < 3 -> no emit
            st.generated_ids.extend([1])
            await d._emit(st, final=False)        # 2 < 3 -> no emit
            self.assertEqual(await _drain(req.stream_queue), [])

            st.generated_ids.extend([2])
            await d._emit(st, final=False)        # 3 >= 3 -> "abc"
            self.assertEqual(await _drain(req.stream_queue), ["abc"])

        asyncio.run(run())

    def test_final_flush_emits_remainder(self):
        async def run():
            arc = FakeArc()
            d = CBInferDaemon("m", arc)
            st, req = _state(arc, _GenCfg(stream_chunk_tokens=5))

            st.generated_ids.extend([0, 1])       # below boundary
            await d._emit(st, final=False)
            self.assertEqual(await _drain(req.stream_queue), [])

            await d._emit(st, final=True)         # forced flush -> "ab"
            self.assertEqual(await _drain(req.stream_queue), ["ab"])

        asyncio.run(run())


class FinalizeSequence(unittest.TestCase):
    def test_finalize_emits_text_then_metrics_then_none(self):
        async def run():
            arc = FakeArc()
            d = CBInferDaemon("m", arc)
            st, req = _state(arc, _GenCfg(stream_chunk_tokens=1))

            st.generated_ids.extend([0, 1, 2])
            await d._emit(st, final=False)        # "abc"
            await d._finalize(st)

            items = await _drain(req.stream_queue)
            self.assertEqual(items[0], "abc")
            self.assertEqual(items[1], {"metrics": {
                "input_token": 7, "new_token": 3, "total_token": 10}})
            self.assertIsNone(items[2])
            self.assertEqual(arc.metrics_calls, [(7, 3)])

        asyncio.run(run())


class AdmitErrorIsolation(unittest.TestCase):
    def test_add_request_failure_closes_only_that_stream(self):
        async def run():
            class BoomArc(FakeArc):
                def add_request(self, rid, cfg):
                    raise RuntimeError("admission boom")

            d = CBInferDaemon("m", BoomArc())
            req = CBRequest(gen_config=_GenCfg(), stream_queue=asyncio.Queue(), int_id=1)
            await d._admit_request(asyncio.get_running_loop(), req)

            self.assertEqual(await _drain(req.stream_queue), [None])
            self.assertNotIn(1, d._active)

        asyncio.run(run())


@unittest.skipUnless(HAS_GENAI, "openvino_genai not installed")
class FullLoopIntegration(unittest.TestCase):
    def test_single_request_full_sequence(self):
        async def run():
            class Handle:
                def __init__(self):
                    self._reads = [[0], [1], [2]]
                    self._i = 0

                def can_read(self):
                    return self._i < len(self._reads)

                def read(self):
                    class O:
                        pass
                    o = O()
                    o.generated_ids = self._reads[self._i]
                    self._i += 1
                    return {0: o}

                def get_status(self):
                    import openvino_genai as genai
                    return (genai.GenerationStatus.RUNNING if self._i < len(self._reads)
                            else genai.GenerationStatus.FINISHED)

            class Arc(FakeArc):
                def add_request(self, rid, cfg):
                    return Handle(), 4

            d = CBInferDaemon("m", Arc())
            d.start()
            req = CBRequest(gen_config=_GenCfg(1), stream_queue=asyncio.Queue(), int_id=1)
            await d.submit(req)

            seen = []
            while True:
                item = await asyncio.wait_for(req.stream_queue.get(), timeout=5)
                if item is None:
                    break
                seen.append(item)
            await d.stop()

            self.assertEqual(seen[0], "a")
            self.assertEqual(seen[-1]["metrics"]["new_token"], 3)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
