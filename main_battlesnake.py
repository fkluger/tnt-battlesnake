import argparse
import os
import bottle

from battlesnake.agent import Agent
from apex.configuration import Configuration

snake: Agent = None
config: Configuration = None


def get_args():
    parser = argparse.ArgumentParser(description='Battlesnake agent')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    return parser.parse_args()


@bottle.route('/static/<path:path>')
def static(path):
    return bottle.static_file(path, root='static/')


@bottle.post('/start')
def start():
    data = bottle.request.json
    game_id = data['game_id']
    board_width = data['width']
    board_height = data['height']

    if snake:
        snake.on_reset()

    return {
        'color': '#FF0000',
        'taunt': '{} ({}x{})'.format(game_id, board_width, board_height),
        'name': 'rl-snake'
    }


@bottle.post('/move')
def move():
    data = bottle.request.json

    return {
        'move': snake.get_direction(data)
    }


@bottle.post('/end')
def end():
    pass


def main():
    global config, snake
    args = get_args()
    application = bottle.default_app()
    config = Configuration(args.config_path)
    snake = Agent(config, args.weights_path)
    bottle.run(application, server='cherrypy', host=os.getenv('IP', '0.0.0.0'), port=os.getenv('PORT', args.port))


if __name__ == '__main__':
    main()
