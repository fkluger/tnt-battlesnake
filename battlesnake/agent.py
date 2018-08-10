import logging

from collections import deque
import numpy as np

from dqn.network import DQN
from apex.configuration import Configuration
from environment.constants import DIRECTIONS
from environment.snake import Snake
from .data_to_state import data_to_state

LOGGER = logging.getLogger('Agent')


class Agent(Snake):

    def __init__(self, config: Configuration, weights_path):
        self.config = config
        self.input_shape = (config.width, config.height, config.stacked_frames)
        self.dqn = DQN(input_shape=self.input_shape, num_actions=3, learning_rate=config.learning_rate)
        self.dqn.online_model.load_weights(weights_path)
        self.dqn.online_model._make_predict_function()
        self.frames = None
        super().__init__([-1, -1])

    def on_reset(self):
        self.head_direction = np.random.choice(DIRECTIONS)
        self.frames = deque(np.zeros([self.config.stacked_frames, self.config.width, self.config.height],
                                     dtype=np.int8), self.config.stacked_frames)

    def get_direction(self, data):

        state = data_to_state(data, self.head_direction)
        self.frames.appendleft(state)
        frames = np.moveaxis(self.frames, 0, -1)
        q_values = np.squeeze(self.dqn.predict(frames))
        best_action = np.argmax(q_values)
        actions = [best_action]
        for i in range(3):
            if i not in actions:
                actions.append(i)
        self.head_direction = self._find_best_action(actions, data)
        return self.head_direction

    def _find_best_action(self, actions, data):
        head = data['you']['body']['data'][0]
        head = [head['x'] + 1, head['y'] + 1]
        directions = [self._get_direction(i) for i in actions]
        for direction in directions:
            next_coord = self._get_next_head(direction, head)
            if not self._check_no_collision(next_coord, data):
                return direction
            else:
                LOGGER.info('Avoided collision! Trying other directions...')
                continue
        LOGGER.info('Giving up!')
        return directions[0]

    def _check_no_collision(self, head, data):
        collision = False
        for s in data['snakes']['data']:
            for body_idx, coord in enumerate(s['body']['data']):
                coord = [coord['x'] + 1, coord['y'] + 1]
                if s['id'] == data['you']['id'] and body_idx == 0:
                    continue
                else:
                    if np.array_equal(head, coord):
                        collision = True
        snake_head_x, snake_head_y = head[0], head[1]
        hit_wall = snake_head_x <= 0 or snake_head_y <= 0 or snake_head_x >= self.config.width - \
            1 or snake_head_y >= self.config.height - 1
        if hit_wall:
            collision = True
        return collision
