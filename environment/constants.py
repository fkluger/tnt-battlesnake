DIRECTIONS = ['up', 'right', 'down', 'left']


class Reward:
    collision = -1
    starve = -1
    nothing = -0.01
    fruit = 0.8

    lost = -1
    won = 1
