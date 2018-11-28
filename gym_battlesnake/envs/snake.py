import random

from gym_battlesnake.envs.constants import Direction


class Snake:
    def __init__(self, head):
        self.health = 100
        self.head_direction = random.choice(list(Direction))
        self.body = [head]
        self.max_length = 3

    def move_head(self, action: int):
        self.health -= 1
        if self.is_dead():
            return
        move_direction = self._get_direction(action)
        next_head = self._get_next_head(move_direction)

        self.body.insert(0, next_head)
        self.head_direction = move_direction

    def move_tail(self, ate_fruit: bool):
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
        self.body.clear()

    def _get_direction(self, action: int):
        if self.head_direction == Direction.up:
            if action == 0:
                return Direction.left
            elif action == 1:
                return Direction.up
            else:
                return Direction.right
        elif self.head_direction == Direction.right:
            if action == 0:
                return Direction.up
            elif action == 1:
                return Direction.right
            else:
                return Direction.down
        elif self.head_direction == Direction.down:
            if action == 0:
                return Direction.right
            elif action == 1:
                return Direction.down
            else:
                return Direction.left
        else:
            if action == 0:
                return Direction.down
            elif action == 1:
                return Direction.left
            else:
                return Direction.up

    def _get_next_head(self, direction: str, head=None):
        if not head:
            head = self.body[0]
        if direction == Direction.up:
            return (head[0], head[1] - 1)
        elif direction == Direction.right:
            return (head[0] + 1, head[1])
        elif direction == Direction.down:
            return (head[0], head[1] + 1)
        else:
            return (head[0] - 1, head[1])
