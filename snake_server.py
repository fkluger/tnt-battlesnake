import os
import sys
import bottle

from battlesnake.rl_agent import RLSnake


snakes = []


@bottle.route('/static/<path:path>')
def static(path):
    return bottle.static_file(path, root='static/')


@bottle.post('/start')
def start():
    data = bottle.request.json
    game_id = data['game_id']
    board_width = data['width']
    board_height = data['height']

    head_url = '%s://%s/static/head.png' % (
        bottle.request.urlparts.scheme,
        bottle.request.urlparts.netloc
    )

    if len(snakes) == 0:
        snakes.append(RLSnake(board_width, board_height, sys.argv[2], sys.argv[3]))

    return {
        'color': '#00FF00',
        'taunt': '{} ({}x{})'.format(game_id, board_width, board_height),
        'head_url': head_url,
        'name': 'rl-snake'
    }


@bottle.post('/move')
def move():
    data = bottle.request.json

    return {
        'move': snakes[0].get_direction(data),
        'taunt': 'Battlesnake!'
    }


# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()
if __name__ == '__main__':
    bottle.run(application, server='cherrypy', host=os.getenv(
        'IP', '0.0.0.0'), port=os.getenv('PORT', sys.argv[1]))
