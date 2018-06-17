DIRECTIONS = ['up', 'right', 'down', 'left']


class Reward:
    collision = -1
    starve = -1
    nothing = -0.01
    fruit = 1
