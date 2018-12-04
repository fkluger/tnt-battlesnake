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
    own_head_up = 110
    own_head_right = 115
    own_head_down = 120
    own_head_left = 125
    own_body = 100
    own_tail = 90
    head = 30
    body = 20
    tail = 10
    fruit = 60
    fruit_color = (215, 115,  85)
    background = (0, 0, 0)
    wall_color = (100, 105, 100)
    snake_color = (80,  140, 215)
