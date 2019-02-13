import json
import os
import random
import bottle

import ray
from battlesnake.api import ping_response, start_response, move_response, end_response
from battlesnake.agent import Agent

agent = None


@bottle.route("/")
def index():
    return """
    Battlesnake documentation can be found at
       <a href="https://docs.battlesnake.io">https://docs.battlesnake.io</a>.
    """


@bottle.route("/static/<path:path>")
def static(path):
    """
    Given a path, return the static file located relative
    to the static folder.

    This can be used to return the snake head URL in an API response.
    """
    return bottle.static_file(path, root="static/")


@bottle.post("/ping")
def ping():
    """
    A keep-alive endpoint used to prevent cloud application platforms,
    such as Heroku, from sleeping the application instance.
    """
    return ping_response()


@bottle.post("/start")
def start():
    global agent
    data = bottle.request.json
    print("Test")
    if agent is None:
        agent = Agent(width=9, height=9, stacked_frames=2)
    agent.on_reset()
    print(json.dumps(data))

    color = "#00529E"

    return start_response(color)


@bottle.post("/move")
def move():
    global agent
    data = bottle.request.json

    print(json.dumps(data))

    direction = agent.get_direction(data)

    return move_response(direction)


@bottle.post("/end")
def end():
    data = bottle.request.json

    print(json.dumps(data))

    return end_response()


# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == "__main__":
    ray.init()
    bottle.run(
        application,
        host=os.getenv("IP", "0.0.0.0"),
        port=os.getenv("PORT", "8080"),
        debug=os.getenv("DEBUG", True),
    )
