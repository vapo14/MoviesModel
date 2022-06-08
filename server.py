import asyncio
import json
import websockets

from model import initialize_model, get_recommendations

# INITIALIZE DATA AND MODEL
df, cosine_sim = initialize_model()


async def handler(websocket, path):

    data = await websocket.recv()

    recommendation_list = get_recommendations(df, data, cosine_sim).values
    rec_dict = {"recommendation_titles": [
        {"title": title} for title in recommendation_list]}
    data = json.dumps(rec_dict)

    await websocket.send(data)


start_server = websockets.serve(handler, "localhost", 8000)


asyncio.get_event_loop().run_until_complete(start_server)

asyncio.get_event_loop().run_forever()
