import argparse
import json
import os
import random
import bottle

from battlesnake.api import ping_response, start_response, move_response, end_response
from battlesnake.agent import Agent

agent_small = None
agent_medium = None
agent_large = None


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
    if data["board"]["width"] == 9:
        agent_small.on_reset()
    elif data["board"]["width"] == 11:
        agent_medium.on_reset()
    else:
        agent_large.on_reset()
    # print(json.dumps(data))

    color = "#00529F"

    return start_response(color)


@bottle.post("/move")
def move():
    global agent
    data = bottle.request.json

    # print(json.dumps(data))

    if data["board"]["width"] == 9:
        direction = agent_small.get_direction(data)
    elif data["board"]["width"] == 11:
        direction = agent_medium.get_direction(data)
    else:
        direction = agent_large.get_direction(data)

    return move_response(direction)


@bottle.post("/end")
def end():
    data = bottle.request.json

    # print(json.dumps(data))

    return end_response()


# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path-small", help="Path to the small checkpoint.", type=str, default=None
    )
    parser.add_argument(
        "path-medium", help="Path to the medium checkpoint.", type=str, default=None
    )
    parser.add_argument(
        "path-large", help="Path to the large checkpoint.", type=str, default=None
    )
    parser.add_argument(
        "--port", help="Port of the web server.", type=str, default="8080"
    )
    args, _ = parser.parse_known_args()
    agent_small = Agent(width=9, height=9, stacked_frames=2, path=args.path_small)
    agent_medium = Agent(width=13, height=13, stacked_frames=2, path=args.path_medium)
    agent_large = Agent(width=21, height=21, stacked_frames=2, path=args.path_large)
    bottle.run(application, host="0.0.0.0", port=args.port, debug=False, quiet=True)
