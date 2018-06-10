import numpy as np


class EnemyActor:

    def __init__(self, actor):
        self.actor = actor

    def act(self, state, snake, snakes, fruits, width, height):
        # TODO: Implement smart actor logic
        action = self.actor.act(state)
        next_head = snake._get_next_head(snake._get_direction(action))
        next_head_x, next_head_y = next_head[0], next_head[1]
        if next_head_x >= 0 and next_head_x < width and next_head_y >= 0 and next_head_y < height:
            return action
        else:
            return np.random.choice(3)
