import random
from collections import deque
import numpy as np

from agents.distributional_dqn import DistributionalDQNAgent
from brains.distributional_dueling_double_dqn import DistributionalDuelingDoubleDQNBrain
from simulator.utils import getDirection, is_coord_on_board, get_next_coord
from .utils import data_to_state


class RLSnake:

    history = []

    def __init__(self, width, height, num_frames, dqn_weights_path):
        # TODO(frederik): Let user pass in experiment path and check whether parameters are correct
        self.width = width + 2
        self.height = height + 2
        self.num_frames = int(num_frames)
        num_quantiles = 20
        input_shape = (self.width + 1, self.height, self.num_frames)
        num_actions = 3
        self.brain = DistributionalDuelingDoubleDQNBrain(num_quantiles=num_quantiles, input_shape=input_shape, num_actions=num_actions)
        self.brain.model.load_weights(dqn_weights_path)
        self.agent = DistributionalDQNAgent(num_quantiles=num_quantiles, brain=self.brain, memory=None, input_shape=input_shape, num_actions=num_actions)
        self.agent.epsilon = 0
        for layer in self.brain.dropout_layers:
            layer.rate = 0
        self.snake_direction = None
        self.frames = None

    def get_last_frames(self, observation):
        '''
        Return a tensor that contains the last num_frames states. The first state will
        be repeated num_frames times.
        '''

        frame = observation
        if self.frames is None:
            self.frames = deque([frame] * self.num_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.moveaxis(np.array(self.frames), 0, -1)

    def get_initial_snake_direction(self):
        return np.random.choice(['up', 'right', 'down', 'left'])

    def get_direction(self, data):

        if self.snake_direction is None:
            self.snake_direction = self.get_initial_snake_direction()
            print('Initial snake direction is {}'.format(self.snake_direction))

        state = data_to_state(data, self.snake_direction)
        frames = self.get_last_frames(state)
        quantiles = self.get_quantiles(frames)
        self.history.append({
            'state': state,
            'quantiles': quantiles,
            'snake_direction': self.snake_direction
        })
        best_action = self.agent.compute_best_action(quantiles)
        actions = [best_action]
        for i in range(3):
            if i not in actions:
                actions.append(i)
        print('Actions: {} with direction {}'.format(actions, self.snake_direction))
        self.snake_direction = self.find_best_action(actions, data)
        print('Choosing direction {}'.format(self.snake_direction))
        return self.snake_direction

    def get_quantiles(self, state):
        quantiles = np.array(self.brain.predict(np.expand_dims(state, 0)))
        return np.swapaxes(quantiles, 0, 1)

    def find_best_action(self, actions, data):
        head = [snake['coords'][0] for snake in data['snakes'] if snake['id'] == data['you']][0]
        head = [head[0] + 1, head[1] + 1]
        directions = [getDirection(i, self.snake_direction) for i in actions]
        for direction in directions:
            next_coord = get_next_coord(head, direction)
            print('Trying to go {}'.format(direction))
            if not self.check_no_collision(next_coord, data):
                return direction
            else:
                print('Avoided collision! Trying other directions...')
                continue
        print('Giving up!')
        return directions[0]

    def check_no_collision(self, head, data):
        collision = False
        for s in data['snakes']:
            for body_idx, coord in enumerate(s['coords']):
                coord = [coord[0] + 1, coord[1] + 1]
                if s['id'] == data['you'] and body_idx == 0:
                    continue
                else:
                    if np.array_equal(head, coord):
                        collision = True
        if not is_coord_on_board(head, self.width, self.height):
            collision = True
        return collision
