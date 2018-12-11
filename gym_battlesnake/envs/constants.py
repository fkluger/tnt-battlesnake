from enum import Enum


class Direction(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


class Reward(Enum):
    collision = -1
    starve = -1
    nothing = -0.01
    fruit = 1

    lost = -10
    won = 10


class Field(Enum):
    own_head_up = 255
    own_head_right = 255
    own_head_down = 255
    own_head_left = 255
    own_body = 180
    own_tail = 160
    head = 70
    body = 40
    tail = 20
    fruit = 130
    fruit_color = (215, 115,  85)
    background = (0, 0, 0)
    wall_color = (100, 105, 100)
    snake_color = (80,  140, 215)
