from collections import deque
from typing import List, Union, Dict

import numpy as np

from gym_battlesnake.envs.snake import Snake
from gym_battlesnake.envs.serialize import serialize


def get_snake_starting_position(width, height, snake_idx):
    positions = {
        0: [2, 2],
        1: [width - 2, height - 2],
        2: [2, height - 2],
        3: [width - 2, 2],
        4: [(width + 2) // 2, 2],
        5: [width - 2, (height + 2) // 2],
        6: [(width + 2) // 2, height - 2],
        7: [2, (height + 2) // 2],
    }
    return positions[snake_idx]


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
        self.fruit_spawn_factor = 2.0
        self.min_fruit_spawn_probability = 0.1
        self.fruit_spawn_probability = 0.1
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

    def move_snakes(self, actions: Union[Dict[str, int], int]):

        if not isinstance(actions, dict):
            actions = {"0": actions}

        for snake_idx, snake in enumerate(self.snakes):
            if str(snake_idx) not in actions:
                snake.die()
                continue
            snake.move_head(actions[str(snake_idx)])

        snake_collided = []
        snake_ate_fruit = []
        snake_starved = []
        snake_won = []
        snake_ate_enemy = []

        for snake_idx, snake in enumerate(self.snakes):
            if str(snake_idx) not in actions:
                continue
            collided, ate_enemy = self._collided(snake, snake.get_head())
            ate_fruit = self._ate_fruit(snake)
            starved = snake.is_dead()

            snake_collided.append(collided)
            snake_ate_fruit.append(ate_fruit)
            snake_starved.append(starved)
            snake_ate_enemy.append(ate_enemy)

            if not collided and not starved:
                snake.move_tail(ate_fruit)
            else:
                snake.die()

        if np.random.random() < self.fruit_spawn_probability:
            self._place_fruits_or_snakes(1, True)
            self.fruit_spawn_probability = self.min_fruit_spawn_probability
        else:
            self.fruit_spawn_probability *= self.fruit_spawn_factor

        for snake_idx, snake in enumerate(self.snakes):
            if str(snake_idx) not in actions:
                continue
            if len(self.snakes) == 1:
                if len(self.snakes[0].body) == (self.width - 2) * (self.height - 2):
                    snake_won.append(True)
                else:
                    snake_won.append(False)
            else:
                snakes_alive = [s for s in self.snakes if s.is_dead() is False]
                snake_won.append(
                    self.snakes[snake_idx].is_dead() is False and len(snakes_alive) == 1
                )

        self._update_state()

        return (
            snake_ate_fruit,
            snake_collided,
            snake_starved,
            snake_won,
            snake_ate_enemy,
        )

    def observe(self, snake_perspective=0):
        return np.moveaxis(
            np.array(self.last_frames_per_snake[snake_perspective]), 0, -1
        )

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
        ate_enemy = False
        for s in self.snakes:
            for s_body_idx, s_body in enumerate(s.body):
                if np.array_equal(snake_head, s_body):
                    if s_body_idx != 0:
                        hit_snake = True
                    else:
                        if snake != s:
                            if len(snake.body) <= len(s.body):
                                hit_snake = True
                            else:
                                ate_enemy = True
        return hit_wall or hit_snake, ate_enemy

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

    def _place_fruits_or_snakes(self, fields: int, is_fruit: bool):
        for i in range(fields):
            field = None
            if is_fruit:
                while not self._is_available(field):
                    field = (
                        np.random.randint(1, self.width - 1),
                        np.random.randint(1, self.height - 1),
                    )
                self.fruits.append(field)
            else:
                field = get_snake_starting_position(self.width, self.height, i)
                self.snakes.append(Snake(field))
