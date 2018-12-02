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
    own_head_up = 210
    own_head_right = 225
    own_head_down = 240
    own_head_left = 255
    own_body = 180
    own_tail = 160
    head = 70
    body = 40
    tail = 20
    fruit = 130
