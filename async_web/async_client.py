import time

from aiohttp import ClientSession, TCPConnector
import asyncio
import sys

from tqdm import tqdm


total_requests = int(sys.argv[1])
max_calls_in_flight = int(sys.argv[2])
port = sys.argv[3]


async def fetch(url, session):
    async with session.get(url) as response:
        return await response.read()


async def _main(url):
    connector = TCPConnector(limit=None)
    async with ClientSession(connector=connector) as session, \
            TaskPool(max_calls_in_flight) as tasks:

        for idx in tqdm(range(total_requests)):
            await tasks.put(fetch(url.format(
                port=port, idx=idx, time=time.time()), session)
            )


def main():
    url = "http://localhost:{port}/nlp/{idx}/{time}"
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_main(url))


class TaskPool(object):
    # From https://medium.com/@cgarciae/making-an-infinite-number-of-requests-with-python-aiohttp-pypeln-3a552b97dc95

    def __init__(self, workers):
        self._semaphore = asyncio.Semaphore(workers)
        self._tasks = set()

    async def put(self, coro):

        await self._semaphore.acquire()

        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task):
        self._tasks.remove(task)
        self._semaphore.release()

    async def join(self):
        await asyncio.gather(*self._tasks)

    async def __aenter__(self):
        return self

    def __aexit__(self, exc_type, exc, tb):
        return self.join()


if __name__ == '__main__':
    main()
