from enum import Enum


class Direction(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


class Reward(Enum):
    nothing = -0.1
    fruit = 1

    lost = -10
    won = 10


class Field(Enum):
    own_head_up = 255
    own_head_right = 245
    own_head_down = 235
    own_head_left = 225
    own_body = 200
    own_tail = 190
    wall = 100
    head = 70
    body = 40
    tail = 20
    fruit = 130
    fruit_color = (215, 115,  85)
    background = (0, 0, 0)
    wall_color = (100, 105, 100)
    snake_color = (80,  140, 215)
