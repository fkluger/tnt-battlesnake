from enum import Enum


class Direction(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


class Reward(Enum):
    nothing = -0.01
    fruit = 1.0
    ate_enemy = 2.0

    lost = -10.0
    won = 5.0


class Field(Enum):
    own_head_up = 255
    own_head_right = 255
    own_head_down = 255
    own_head_left = 255
    own_body = 200
    own_tail = 190
    wall = 100
    head = 70
    body = 40
    tail = 20
    fruit = 130
