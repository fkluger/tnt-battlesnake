from collections import deque
from typing import List, Union

import numpy as np

from gym_battlesnake.envs.snake import Snake
from gym_battlesnake.envs.serialize import serialize


class State:
    def __init__(
        self,
        width: int,
        height: int,
        num_snakes: int,
        num_fruits: int,
        stacked_frames: int = 1,
    ):
        self.width = width
        self.height = height
        self.fruits = []
        self.snakes = []
        self.stacked_frames = stacked_frames
        self.last_frames_per_snake = [
            deque(
                np.zeros(
                    [self.stacked_frames, self.width, self.height], dtype=np.uint8
                ),
                self.stacked_frames,
            )
            for snake_idx in range(num_snakes)
        ]
        self._place_fruits_or_snakes(num_fruits, True)
        self._place_fruits_or_snakes(num_snakes, False)
        self._update_state()

    def move_snakes(self, actions: Union[List[int], int]):

        if not isinstance(actions, list):
            actions = [actions]

        for snake_idx, snake in enumerate(self.snakes):
            snake.move_head(actions[snake_idx])

        snake_collided = False
        snake_ate_fruit = False
        snake_starved = False
        snake_won = False

        fruits_eaten = 0

        for snake_idx, snake in enumerate(self.snakes):
            collided = self._collided(snake, snake.get_head())
            ate_fruit = self._ate_fruit(snake)
            starved = snake.is_dead()

            if ate_fruit:
                fruits_eaten += 1

            if snake_idx == 0:
                snake_collided = collided
                snake_ate_fruit = ate_fruit
                snake_starved = starved

            if not collided and not starved:
                snake.move_tail(ate_fruit)
            else:
                snake.die()

        self._place_fruits_or_snakes(fruits_eaten, True)

        if len(self.snakes) == 1:
            snake_won = False
        else:
            snakes_alive = [s for s in self.snakes if s.is_dead() is False]
            snake_won = self.snakes[0].is_dead() is False and len(snakes_alive) == 1

        self._update_state()

        return snake_ate_fruit, snake_collided, snake_starved, snake_won

    def observe(self, snake_perspective=0):
        return np.array(self.last_frames_per_snake[snake_perspective])

    def _update_state(self):
        for snake_idx, state in enumerate(self.snakes):
            state = serialize(
                width=self.width,
                height=self.height,
                snakes=self.snakes,
                fruits=self.fruits,
                own_snake_index=snake_idx,
            )
            self.last_frames_per_snake[snake_idx].appendleft(state)

    def _collided(self, snake: Snake, snake_head: List[int]):
        snake_head_x, snake_head_y = snake_head[0], snake_head[1]
        hit_wall = (
            True
            if (
                snake_head_x <= 0
                or snake_head_y <= 0
                or snake_head_x >= (self.width - 1)
                or snake_head_y >= (self.height - 1)
            )
            else False
        )
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

    def _ate_fruit(self, snake: Snake):
        ate_fruit = False
        snake_head = snake.body[0]
        for fruit in self.fruits:
            if np.array_equal(snake_head, fruit):
                self.fruits.remove(fruit)
                ate_fruit = True
        return ate_fruit

    def _is_available(self, field: List[int]):
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

    def _place_fruits_or_snakes(self, fields: List[int], is_fruit: bool):
        # Leave one row/column for snakes to prevent instant-death episodes
        padding = 1 if is_fruit else 2
        for _ in range(fields):
            field = None
            while not self._is_available(field):
                field = (
                    np.random.randint(padding, self.width - padding),
                    np.random.randint(padding, self.height - padding),
                )
            if is_fruit:
                self.fruits.append(field)
            else:
                self.snakes.append(Snake(field))
