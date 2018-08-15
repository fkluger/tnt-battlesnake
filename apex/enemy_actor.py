import numpy as np

from environment.snake import Snake
from environment.state import State
from apex.actor import Actor


class EnemyActor:
    def __init__(self, actor: Actor):
        self.actor = actor

    def act(self, state: State, snake_idx: int):
        snake: Snake = state.snakes[snake_idx]
        default_action, greedy = self.actor.act(state.observe(snake_idx))
        possible_actions = [default_action]
        possible_actions.extend([i for i in range(3) if i != default_action])
        # Try all actions starting with the default action, chosen by the actor
        for action in possible_actions:
            next_head = snake._get_next_head(snake._get_direction(action))
            collided = state._collided(snake, next_head)
            if not collided:
                return action
        # Give up and return some random action
        return action
