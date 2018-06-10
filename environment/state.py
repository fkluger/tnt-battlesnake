import numpy as np

from .snake import Snake
from .state_serializer import StateSerializer


class State:

    def __init__(self, width, height, snakes, fruits):
        self.width = width
        self.height = height
        self.fruits = []
        self.snakes = []
        self._place_fruits_or_snakes(fruits, True)
        self._place_fruits_or_snakes(snakes, False)
        self.serializer = StateSerializer()

    def move_snakes(self, actions):

        for snake_idx, snake in enumerate(self.snakes):
            snake.move_head(actions[snake_idx])

        snake_collided = False
        snake_ate_fruit = False
        snake_starved = False
        snake_won = False

        for snake_idx, snake in enumerate(self.snakes):
            collided = self._collided(snake)
            ate_fruit = self._ate_fruit(snake)
            starved = snake.is_dead()

            if snake_idx == 0:
                snake_collided = collided
                snake_ate_fruit = ate_fruit
                snake_starved = starved

            if not collided and not starved:
                snake.move_tail(ate_fruit)
            else:
                snake.health = 0

        if len(self.snakes) == 1:
            snake_won = False
        else:
            snakes_alive = [s for s in self.snakes if s.is_dead() is False]
            snake_won = self.snakes[0].is_dead() is False and len(snakes_alive) == 1

        return snake_ate_fruit, snake_collided, snake_starved, snake_won

    def observe(self, snake_perspective=0):
        return self.serializer.serialize(snake_perspective, self)

    def _collided(self, snake):
        snake_head = snake.get_head()
        snake_head_x, snake_head_y = snake_head[0], snake_head[1]
        hit_wall = snake_head_x <= 0 or snake_head_y <= 0 or snake_head_x >= self.width - 1 or snake_head_y >= self.height - 1
        hit_snake = False
        for s in self.snakes:
            for s_body_idx, s_body in enumerate(s.body):
                if np.array_equal(snake_head, s_body):
                    if s_body_idx != 0:
                        hit_snake = True
                    else:
                        if snake != s and len(snake.body) <= len(s.body):
                            hit_snake = True
        return hit_wall or hit_snake

    def _ate_fruit(self, snake):
        ate_fruit = False
        snake_head = snake.body[0]
        for fruit in self.fruits:
            if np.array_equal(snake_head, fruit):
                self.fruits.remove(fruit)
                ate_fruit = True
        return ate_fruit

    def _is_available(self, field):
        if field is None:
            return False

        available = True
        for fruit_field in self.fruits:
            if np.array_equal(field, fruit_field):
                available = False
        for snake in self.snakes:
            for snake_field in snake.body:
                if np.array_equal(field, snake_field):
                    available = False
        return available

    def _place_fruits_or_snakes(self, fields, is_fruit):
        for _ in range(fields):
            field = None
            while not self._is_available(field):
                field = [
                    np.random.randint(1, self.width),
                    np.random.randint(1, self.height)
                ]
            if is_fruit:
                self.fruits.append(field)
            else:
                self.snakes.append(Snake(field))
