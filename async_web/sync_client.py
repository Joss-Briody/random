import sys
import time

import requests
from tqdm import tqdm

total_requests = int(sys.argv[1])
port = sys.argv[2]


def main():
    URL = "http://localhost:{port}/nlp/{idx}/{time}"
    for idx in tqdm(range(total_requests)):
        _ = requests.get(  # noqa
            url=URL.format(port=port, idx=idx, time=time.time())
        )


if __name__ == '__main__':
    main()
