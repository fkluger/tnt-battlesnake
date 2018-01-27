import numpy as np
from collections import deque
from keras.models import load_model
from keras.optimizers import RMSprop

from brains.dueling_double_dqn import DuelingDoubleDQNBrain
from simulator.utils import getDirection, is_coord_on_board, get_next_coord
from .utils import data_to_state


class RLSnake:

    def __init__(self, width, height, num_frames, dqn_weights_path):
        self.width = width + 2
        self.height = height + 2
        self.brain = DuelingDoubleDQNBrain(input_shape=(self.width + 1, self.height, 1), num_actions=3)
        self.brain.model = load_model(dqn_weights_path, compile=False)
        self.brain.model.compile(optimizer=RMSprop(lr=self.brain.learning_rate), loss='mse')
        self.snake_direction = None
        self.frames = None
        self.num_frames = int(num_frames)

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

    def get_initial_snake_direction(self, data):
        snake = [snake for snake in data['snakes'] if snake['id'] == data['you']][0]
        head = snake['coords'][0]
        neck = snake['coords'][1]
        if head[0] < neck[0]:
            return 'left'
        elif head[0] > neck[0]:
            return 'right'
        elif head[1] < neck[1]:
            return 'down'
        else:
            return 'up'

    def get_direction(self, data):

        if self.snake_direction is None:
            self.snake_direction = self.get_initial_snake_direction(data)
            print('Initial snake direction is {}'.format(self.snake_direction))

        state = data_to_state(data, self.snake_direction)
        frames = self.get_last_frames(state)
        q_values = self.brain.predict(frames[np.newaxis, ...])[0]
        print('Q-Values: {}'.format(q_values))
        self.snake_direction = self.find_best_action(q_values, data)
        print('Choosing direction {}'.format(self.snake_direction))
        return self.snake_direction

    def find_best_action(self, q_values, data):
        head = [snake['coords'][0] for snake in data['snakes'] if snake['id'] == data['you']][0]
        head = [head[0] + 1, head[1] + 1]
        actions = np.argsort(q_values)[::-1]
        directions = [getDirection(i, self.snake_direction) for i in actions]
        for direction in directions:
            next_coord = get_next_coord(head, direction)
            if is_coord_on_board(next_coord, self.width, self.height):
                return direction
        return directions[0]
