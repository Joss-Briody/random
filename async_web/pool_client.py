import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from contexttimer import Timer

total_requests = int(sys.argv[1])
max_calls_in_flight = int(sys.argv[2])
port = sys.argv[3]


def main():
    URL = "http://localhost:{port}/nlp/{idx}/{time}"
    with Timer() as t:
        with ThreadPoolExecutor(max_workers=max_calls_in_flight) as exe:
            _ = exe.map(requests.get, (  # noqa
                URL.format(port=port, idx=idx, time=time.time()) for idx in range(total_requests))
            )

    print(f'Processed {total_requests/t.elapsed} requests per second')


if __name__ == '__main__':
    main()
