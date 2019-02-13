import logging

from collections import deque
import numpy as np

from ray.rllib.agents.dqn import DQNAgent

from gym_battlesnake.envs.snake import Snake
from gym_battlesnake.envs.constants import Direction
from .data_to_state import data_to_state
from train import get_agent_config

LOGGER = logging.getLogger("Agent")


class Agent(Snake):
    def __init__(self, width: int, height: int, stacked_frames: int, path: str = None):
        self.width = width
        self.height = height
        self.stacked_frames = stacked_frames
        config = get_agent_config(
            width=width, height=height, stacked_frames=stacked_frames, num_snakes=1
        )
        config["num_workers"] = 0
        config["num_envs_per_worker"] = 1
        del config["multiagent"]
        self.dqn = DQNAgent(config=config, env="battlesnake")
        if path:
            self.dqn.restore(path)

    def on_reset(self):
        self.head_direction = Direction.up
        self.frames = deque(
            np.zeros([self.stacked_frames, self.width, self.height], dtype=np.uint8),
            self.stacked_frames,
        )

    def get_direction(self, data):
        state = data_to_state(self.width, self.height, data, self.head_direction)
        self.frames.appendleft(state)
        observation = np.moveaxis(self.frames, 0, -1)
        best_action = self.dqn.compute_action(observation)
        actions = [best_action]
        for i in range(3):
            if i not in actions:
                actions.append(i)
        self.head_direction = self._find_best_action(actions, data)
        if self.head_direction == Direction.up:
            return "up"
        elif self.head_direction == Direction.left:
            return "left"
        elif self.head_direction == Direction.down:
            return "down"
        else:
            return "right"

    def _find_best_action(self, actions, data):
        head = data["you"]["body"][0]
        head = [head["x"] + 1, head["y"] + 1]
        directions = [self._get_direction(i) for i in actions]
        for direction in directions:
            next_coord = self._get_next_head(direction, head)
            if not self._check_no_collision(next_coord, data):
                return direction
            else:
                LOGGER.info("Avoided collision! Trying other directions...")
                continue
        LOGGER.info("Giving up!")
        return directions[0]

    def _check_no_collision(self, head, data):
        collision = False
        for s in data["board"]["snakes"]:
            for body_idx, coord in enumerate(s["body"]):
                coord = [coord["x"] + 1, coord["y"] + 1]
                if s["id"] == data["you"]["id"] and body_idx == 0:
                    continue
                else:
                    if np.array_equal(head, coord):
                        collision = True
        snake_head_x, snake_head_y = head[0], head[1]
        hit_wall = (
            snake_head_x <= 0
            or snake_head_y <= 0
            or snake_head_x >= self.width - 1
            or snake_head_y >= self.height - 1
        )
        if hit_wall:
            collision = True
        return collision
