#!/usr/bin/env bash

ASYNC_SERVER_PORT=5001
SYNC_SERVER_PORT=5002
NUM_THREADS=100
POOL_SIZE=1000

#python async_web_server.py && python sync_web_server.py &

python sync_client.py 1000 ${SYNC_SERVER_PORT}
python sync_client.py 1000 ${ASYNC_SERVER_PORT}


python pool_client.py 1000 ${NUM_THREADS} ${SYNC_SERVER_PORT}
python pool_client.py 1000 ${NUM_THREADS} ${ASYNC_SERVER_PORT}


python async_client.py 100000 ${POOL_SIZE} ${ASYNC_SERVER_PORT}
python async_client.py 100000 ${POOL_SIZE} ${SYNC_SERVER_PORT}
