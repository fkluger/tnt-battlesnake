import numpy as np

from .constants import DIRECTIONS


class Snake:

    def __init__(self, head):
        self.health = 100
        self.head_direction = np.random.choice(DIRECTIONS)
        self.body = [head]
        self.max_length = 3

    def move_head(self, action):
        self.health -= 1
        if self.is_dead():
            return
        move_direction = self._get_direction(action)
        next_head = self._get_next_head(move_direction)

        self.body.insert(0, next_head)
        self.head_direction = move_direction

    def move_tail(self, ate_fruit):
        if ate_fruit:
            self.max_length += 1
            self.health = 100
        if len(self.body) > self.max_length:
            self.body.pop()

    def is_dead(self):
        return self.health <= 0

    def get_head(self):
        return self.body[0]

    def die(self):
        self.health = 0
        self.body = []

    def _get_direction(self, action):
        if self.head_direction == 'up':
            if action == 0:
                return 'left'
            elif action == 1:
                return 'up'
            else:
                return 'right'
        elif self.head_direction == 'right':
            if action == 0:
                return 'up'
            elif action == 1:
                return 'right'
            else:
                return 'down'
        elif self.head_direction == 'down':
            if action == 0:
                return 'right'
            elif action == 1:
                return 'down'
            else:
                return 'left'
        else:
            if action == 0:
                return 'down'
            elif action == 1:
                return 'left'
            else:
                return 'up'

    def _get_next_head(self, direction):
        head = self.body[0]
        if direction == 'up':
            return [head[0], head[1] - 1]
        elif direction == 'right':
            return [head[0] + 1, head[1]]
        elif direction == 'down':
            return [head[0], head[1] + 1]
        else:
            return [head[0] - 1, head[1]]
