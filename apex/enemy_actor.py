import numpy as np


class EnemyActor:

    def __init__(self, actor):
        self.actor = actor

    def act(self, state, snake, snakes, fruits, width, height):
        # TODO: Implement smart actor logic
        action = self.actor.act(state)
        next_head = snake._get_next_head(snake._get_direction(action))
        next_head_x, next_head_y = next_head[0], next_head[1]
        hit_wall = next_head_x >= 0 and next_head_x < width and next_head_y >= 0 and next_head_y < height
        hit_snake = False
        for s in snakes:
            if s == snake:
                continue
            for s_body_idx, s_body = enumerate(s.body):
                if np.array_equal(next_head, s_body):
                    if s_body_idx != 0:
                        hit_snake = True
                    else:
                        if len(snake.body) < len(s.body):
                            hit_snake = True
        if not hit_snake and not hit_wall:
            return action
        else:
            possible_actions = np.arange(3)
            possible_actions.remove(action)
            return np.random.choice(possible_actions)

