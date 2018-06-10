import numpy as np

from .constants import DIRECTIONS


class Snake:

    health = 100

    def __init__(self, head):
        self.head_direction = np.random.choice(DIRECTIONS)
        self.body = [head]

    def move_head(self, action):
        self.health -= 1
        if self.is_dead():
            return
        move_direction = self._get_direction(action)
        next_head = self._get_next_head(move_direction)

        self.body.insert(0, next_head)

    def move_tail(self, ate_fruit):
        if ate_fruit:
            self.health = 100
        if not ate_fruit and len(self.body) > 3:
            self.body.pop()

    def is_dead(self):
        return self.health <= 0

    def get_head(self):
        return self.body[0]

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
