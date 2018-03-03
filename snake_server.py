import os
import sys
import bottle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from battlesnake.rl_agent import RLSnake
from simulator.utils import getDirection, DIRECTIONS


snakes = []


@bottle.route('/static/<path:path>')
def static(path):
    return bottle.static_file(path, root='static/')


def save_episode(history):
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.17,10.79))
    ax1.set_axis_off()
    ax1.set_title('State')
    ax2.grid()
    ax2.set_title('Mean Q-Values')
    # ax3.grid()
    fig.tight_layout()
    images = []
    for step in history:
        state, quantiles = step['state'], step['quantiles']
        state = np.transpose(state)
        img = ax1.imshow(state[:, 0:-1], animated=True)
        quantiles = np.squeeze(quantiles)
        mean = np.mean(quantiles, axis=1)
        # variance = np.var(quantiles, axis=1)
        possible_directions = [getDirection(a, step['snake_direction']) for a in range(3)]
        y = np.zeros(4)
        # yerr = np.zeros(4)
        for a in range(4):
            try:
                idx = possible_directions.index(DIRECTIONS[a])
                y[a] = mean[idx]
                # yerr[a] = variance[idx]
            except ValueError:
                y[a] = 0
                # yerr[a] = 0
        x = DIRECTIONS
        barplot = ax2.bar(x, y, color=['red', 'green', 'blue', 'yellow'])
        # barplot2 = ax3.bar(x, yerr)
        mean0, mean1, mean2, mean3 = barplot.patches
        # variance0, variance1, variance2, variance3 = barplot2.patches
        # images.append([img, mean0, mean1, mean2, mean3, variance0, variance1, variance2, variance3])
        images.append([img, mean0, mean1, mean2, mean3])
    ani = animation.ArtistAnimation(fig, images, interval=100, repeat=False, blit=True)
    ani.save('episode.mp4')
    plt.close()


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
    else:
        s = snakes[0]
        # if s.history:
        #     save_episode(s.history)
        #     s.history = []

    return {
        'color': '#FF0000',
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
