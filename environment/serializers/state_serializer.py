from collections import deque

import numpy as np


class Field:
    own_head_up = -42
    own_head_right = -42
    own_head_down = -42
    own_head_left = -42
    own_body = 32
    own_tail = 33
    head = 31
    body = 32
    tail = 33
    fruit = 42


class StateSerializer:

    def __init__(self, snakes, width, height, stacked_frames):
        self.snakes = snakes
        self.width = width
        self.height = height
        self.stacked_frames = stacked_frames
        self.on_reset()

    def on_reset(self):
        self.last_frames_per_snake = [deque(np.zeros(
            [self.stacked_frames, self.width, self.height], dtype=np.int8), self.stacked_frames) for snake_idx in range(self.snakes)]

    def serialize(self, snake_perspective, state):
        if snake_perspective == 0:
            for snake_idx in range(self.snakes):
                current_state = np.zeros([state.width, state.height], dtype=np.int8)
                for x in range(state.width):
                    for y in range(state.height):
                        if x == 0 or y == 0 or x == state.width - 1 or y == state.height - 1:
                            current_state[x, y] = Field.body
                for snake_index, snake in enumerate(state.snakes):
                    if snake.is_dead():
                        continue
                    snake_length = len(snake.body)
                    for body_idx, [x, y] in enumerate(snake.body):
                        # Snake at position 0 is the agent
                        if snake_index == snake_idx:
                            if body_idx == 0:
                                if snake.head_direction == 'up':
                                    current_state[x, y] = Field.own_head_up
                                elif snake.head_direction == 'right':
                                    current_state[x, y] = Field.own_head_right
                                elif snake.head_direction == 'down':
                                    current_state[x, y] = Field.own_head_down
                                else:
                                    current_state[x, y] = Field.own_head_left
                            else:
                                current_state[x, y] = Field.own_tail if body_idx == snake_length - 1 else Field.own_body
                        else:
                            current_state[x, y] = Field.head if body_idx == 0 else Field.tail if body_idx == snake_length - 1 else Field.body
                for [x, y] in state.fruits:
                    current_state[x, y] = Field.fruit
                self.last_frames_per_snake[snake_idx].appendleft(current_state)
        frames = np.moveaxis(self.last_frames_per_snake[snake_perspective], 0, -1)
        return frames
