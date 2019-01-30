from typing import List
import math

import numpy as np

from gym_battlesnake.envs.constants import Field, Direction
from gym_battlesnake.envs.snake import Snake


def serialize_window(
    width: int,
    height: int,
    window_width: int,
    window_height: int,
    snakes: List[Snake],
    # [x, y]
    fruits: List[List[int]],
    own_snake_index: int = 0,
):
    current_window = np.zeros(shape=(window_width, window_height), dtype=np.uint8)
    if snakes[own_snake_index].is_dead():
        return current_window
    center_x, center_y = snakes[own_snake_index].get_head()
    walls = (
        [(0, y) for y in range(height)]
        + [(x, height - 1) for x in range(width)]
        + [(width - 1, y) for y in range(height)]
        + [(x, 0) for x in range(width)]
    )

    half_window_width = math.floor(window_width / 2.0)
    half_window_height = math.floor(window_height / 2.0)

    def is_in_window(x, y):
        return (
            x >= max(center_x - half_window_width, 0)
            and x < min(center_x + half_window_width, width)
            and y >= max(center_y - half_window_height, 0)
            and y < min(center_y + half_window_height, height)
        )

    def to_window_coordinates(x, y):
        return (x - center_x + half_window_width, y - center_y + half_window_height)

    for (x, y) in walls:
        if is_in_window(x, y):
            transformed = to_window_coordinates(x, y)
            current_window[transformed] = Field.wall.value
    for (x, y) in fruits:
        if is_in_window(x, y):
            transformed = to_window_coordinates(x, y)
            current_window[transformed] = Field.fruit.value
    for snake_index, snake in enumerate(snakes):
        if snake.is_dead():
            continue
        snake_length = len(snake.body)
        for body_idx, (x, y) in enumerate(snake.body):
            if not is_in_window(x, y):
                continue
            transformed_coordinates = to_window_coordinates(x, y)
            # Snake at position 0 is the agent
            if snake_index == own_snake_index:
                if body_idx == 0:
                    if snake.head_direction == Direction.up:
                        current_window[
                            transformed_coordinates
                        ] = Field.own_head_up.value
                    elif snake.head_direction == Direction.right:
                        current_window[
                            transformed_coordinates
                        ] = Field.own_head_right.value
                    elif snake.head_direction == Direction.down:
                        current_window[
                            transformed_coordinates
                        ] = Field.own_head_down.value
                    else:
                        current_window[
                            transformed_coordinates
                        ] = Field.own_head_left.value
                else:
                    current_window[transformed_coordinates] = (
                        Field.own_tail.value
                        if body_idx == snake_length - 1
                        else Field.own_body.value
                    )
            else:
                current_window[transformed_coordinates] = (
                    Field.head.value
                    if body_idx == 0
                    else Field.tail.value
                    if body_idx == snake_length - 1
                    else Field.body.value
                )
    return current_window
