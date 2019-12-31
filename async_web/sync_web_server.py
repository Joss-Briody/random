import random
import time

from flask import Flask, jsonify

app = Flask(__name__)

PORT = 5002
MULT = 0.1


@app.route('/nlp/<idx>/<send_time>')
def get_handler(idx, send_time):
    r = MULT * random.random()
    print(f'Got a request {idx}, sleeping {r} seconds')
    time.sleep(r)
    return jsonify({'wait': r, 'idx': idx, 'start': send_time})


if __name__ == '__main__':
    app.run(port=PORT)
