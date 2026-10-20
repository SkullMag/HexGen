"""Network progress and reply confirmation must survive queued inference."""
import ast
import asyncio
import json
import sys
import threading
import types
import unittest
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1] / 'utils'
sys.path.insert(0, str(UTILS))
from serial_inference import SerialInference


class WorkerDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_network_progress_while_inference_stays_serial(self):
        worker = SerialInference()
        release, started = threading.Event(), threading.Event()
        events = []

        def infer(i):
            events.append(('start', i))
            started.set()
            if i == 0:
                self.assertTrue(release.wait(2), 'Network loop was starved')
            events.append(('finish', i))
            return i

        try:
            first = asyncio.create_task(worker.run(infer, 0))
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            second = asyncio.create_task(worker.run(infer, 1))
            await asyncio.sleep(.01)
            self.assertEqual(events, [('start', 0)])
            release.set()
            self.assertEqual(await asyncio.gather(first, second), [0, 1])
            self.assertEqual(events, [('start', 0), ('finish', 0), ('start', 1), ('finish', 1)])
        finally:
            release.set()
            worker.close()

    async def test_failed_inference_does_not_disable_next_request(self):
        worker = SerialInference()
        def fail():
            raise ValueError('inference failed')
        try:
            with self.assertRaisesRegex(ValueError, 'inference failed'):
                await worker.run(fail)
            self.assertEqual(await worker.run(lambda: 'next result'), 'next result')
        finally:
            worker.close()

    async def test_reply_waits_for_transport_confirmation(self):
        # Load the real reply method without requiring a CUDA installation.
        tree = ast.parse((UTILS / '_base_rank_based.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InferenceWorker')
        method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'reply')
        scope = {'json': json}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<reply>', 'exec'), scope)
        published, confirmed = asyncio.Event(), asyncio.Event()

        class Transport:
            async def publish(self, subject, data):
                self.payload = json.loads(data)
                published.set()
            async def flush(self, timeout):
                await confirmed.wait()

        transport = Transport()
        task = asyncio.create_task(scope['reply'](types.SimpleNamespace(nc=transport),
                                                  types.SimpleNamespace(reply='request.inbox'), ['result', 3.4]))
        await asyncio.wait_for(published.wait(), 1)
        await asyncio.sleep(0)
        self.assertFalse(task.done(), 'A buffered reply was treated as sent')
        confirmed.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(transport.payload, ['result', 3.4])


if __name__ == '__main__':
    unittest.main()
