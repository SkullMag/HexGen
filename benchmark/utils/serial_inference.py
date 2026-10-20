"""Run blocking inference serially without blocking network I/O."""
import asyncio
from concurrent.futures import ThreadPoolExecutor


class SerialInference:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gpu-inference')

    async def run(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    def close(self):
        self.executor.shutdown(wait=True)
