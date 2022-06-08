import asyncio

import websockets

from model import initialize_model, get_recommendations

df, cosine_sim = initialize_model()

# create handler for each connection


async def handler(websocket, path):

    data = await websocket.recv()

    recommendation_list = get_recommendations(df, data, cosine_sim)

    reply = "\n".join(recommendation_list.values)

    await websocket.send(reply)


start_server = websockets.serve(handler, "localhost", 8000)


asyncio.get_event_loop().run_until_complete(start_server)

asyncio.get_event_loop().run_forever()
