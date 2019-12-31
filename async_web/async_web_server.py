import random
import asyncio

from aiohttp import web

app = web.Application()
PORT = 5001
MULT = 0.1

routes = web.RouteTableDef()


@routes.get('/nlp/{idx}/{time}')
async def get_handler(request):
    r = MULT * random.random()
    print(f'Got a request {request.match_info["idx"]}, sleeping {r:.2f} seconds')
    await asyncio.sleep(r)
    return web.json_response({'wait': r, 'idx': request.match_info["idx"], 'start': request.match_info["time"]})


app.add_routes(routes)
web.run_app(app, port=PORT)
