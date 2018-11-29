from enum import Enum


class Direction(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


class Reward(Enum):
    collision = -1
    starve = -1
    nothing = -0.1
    fruit = 0.8

    lost = -1
    won = 1


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
