import os
from typing import List, Union

import gym
from gym import spaces
import numpy as np

from gym_battlesnake.envs.state import State
from gym_battlesnake.envs.constants import Reward


class BattlesnakeEnv(gym.Env):

    """
    Base class for different Battlesnake gym environments.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, width: int, height: int, num_fruits: int = 1, sparse_rewards: bool = False
    ):

        self.width = width
        self.height = height

        self.sparse_rewards = sparse_rewards
        self.num_fruits = num_fruits
        self.num_snakes = 1
        self.state = State(
            width=self.width,
            height=self.height,
            num_snakes=self.num_snakes,
            num_fruits=self.num_fruits,
        )
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, self.width, self.height), dtype=np.uint8
        )

    def reset(self):
        self.state = State(
            width=self.width,
            height=self.height,
            num_snakes=self.num_snakes,
            num_fruits=self.num_fruits,
        )
        if self.num_snakes == 1:
            return self.state.observe()
        else:
            return dict(
                zip(
                    range(self.num_snakes),
                    [self.state.observe(i) for i in range(self.num_snakes)],
                )
            )

    def step(self, action: Union[List[int], int]):

        fruit_eaten, collided, starved, won = self.state.move_snakes(action)

        reward, terminal = self._evaluate_reward(fruit_eaten, collided, starved, won)

        terminal = terminal or won

        if terminal:
            next_state = None
        else:
            if self.num_snakes == 1:
                next_state = self.state.observe()
            else:
                next_state = dict(
                    zip(
                        range(self.num_snakes),
                        [self.state.observe(i) for i in range(self.num_snakes)],
                    )
                )

        return next_state, reward, terminal, {}

    def render(self, mode="human"):
        print(self.state.observe())

    def _evaluate_reward(
        self, fruit_eaten: bool, collided: bool, starved: bool, won: bool
    ):
        terminal = False
        reward = Reward.nothing.value
        if self.sparse_rewards:
            if collided or starved:
                reward = Reward.lost.value
                terminal = True
            elif won:
                reward = Reward.won.value
                terminal = True
        else:
            if collided:
                terminal = True
                reward = Reward.collision.value
            else:
                if won:
                    reward = Reward.won.value
                elif fruit_eaten:
                    reward = Reward.fruit.value
                elif starved:
                    terminal = True
                    reward = Reward.starve.value
        return reward, terminal
