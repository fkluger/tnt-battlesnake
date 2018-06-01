class Reward:
    collision = -1
    won = 10
    starve = -10
    nothing = 0
    fruit = 1
    moved_to_fruit = 0.01


class Field:
    own_head_up = -42
    own_head_right = -43
    own_head_down = -44
    own_head_left = -45
    own_body = 32
    own_tail = 33
    head = 31
    body = 32
    tail = 33
    fruit = 42
