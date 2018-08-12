import numpy as np


class EnemyActor:
    def __init__(self, actor):
        self.actor = actor

    def act(self, state, snake_idx):
        snake = state.snakes[snake_idx]
        action, greedy = self.actor.act(state.observe(snake_idx))
        next_head = snake._get_next_head(snake._get_direction(action))
        collided = state._collided(snake, next_head)
        if not collided:
            return action
        else:
            possible_actions = [i for i in range(3) if i != action]
            return np.random.choice(possible_actions)
