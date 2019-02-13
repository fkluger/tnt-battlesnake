from typing import List

import numpy as np

from gym_battlesnake.envs.constants import Field, Direction
from gym_battlesnake.envs.snake import Snake


def serialize(
    width: int,
    height: int,
    snakes: List[Snake],
    fruits: List[List[int]],
    own_snake_index: int = 0,
):
    current_state = np.zeros((width, height), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                current_state[x, y] = Field.body.value
    snake_direction = None
    for snake_index, snake in enumerate(snakes):
        if snake_index == own_snake_index:
            current_state[0, height - 1] = snake.health
        else:
            current_state[snake_index, 0] = snake.health
        if snake.is_dead():
            continue
        snake_length = len(snake.body)
        for body_idx, [x, y] in enumerate(snake.body):
            if snake_index == own_snake_index:
                if snake_direction is None:
                    snake_direction = snake.head_direction
                if body_idx == 0:
                    if snake.head_direction == Direction.up:
                        current_state[x, y] = Field.own_head_up.value
                    elif snake.head_direction == Direction.right:
                        current_state[x, y] = Field.own_head_right.value
                    elif snake.head_direction == Direction.down:
                        current_state[x, y] = Field.own_head_down.value
                    else:
                        current_state[x, y] = Field.own_head_left.value
                else:
                    current_state[x, y] = (
                        Field.own_tail.value
                        if body_idx == snake_length - 1
                        else Field.own_body.value
                    )
            else:
                current_state[x, y] = (
                    Field.head.value
                    if body_idx == 0
                    else Field.tail.value
                    if body_idx == snake_length - 1
                    else Field.body.value
                )
    for [x, y] in fruits:
        current_state[x, y] = Field.fruit.value
    if snake_direction == Direction.left:
        current_state = np.rot90(current_state, k=3)
    elif snake_direction == Direction.down:
        current_state = np.rot90(current_state, k=2)
    elif snake_direction == Direction.right:
        current_state = np.rot90(current_state, k=1)
    return current_state
