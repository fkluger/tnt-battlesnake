import os
from typing import Dict, List, Union

import numpy as np
from gym import spaces
from ray.rllib.env import MultiAgentEnv

from gym_battlesnake.envs.constants import Reward
from gym_battlesnake.envs.state import State


class BattlesnakeEnv(MultiAgentEnv):

    """
    Base class for different Battlesnake gym environments.
    """

    def __init__(self, width: int, height: int, num_snakes: int, stacked_frames: int):

        self.width = width
        self.height = height

        self.num_fruits = num_snakes
        self.num_snakes = num_snakes
        self.stacked_frames = stacked_frames
        self.state = State(
            width=self.width,
            height=self.height,
            num_snakes=self.num_snakes,
            num_fruits=self.num_fruits,
            stacked_frames=self.stacked_frames,
        )
        self.action_space = spaces.Discrete(3)
        self.obs_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.width, self.height, self.stacked_frames),
            dtype=np.uint8,
        )
        if num_snakes == 1:
            self.observation_space = self.obs_space
        else:
            self.observation_space = spaces.Dict(
                dict(
                    zip(
                        map(str, range(num_snakes)),
                        [self.obs_space for _ in range(num_snakes)],
                    )
                )
            )

    def reset(self):
        self.state = State(
            width=self.width,
            height=self.height,
            num_snakes=self.num_snakes,
            num_fruits=self.num_fruits,
            stacked_frames=self.stacked_frames,
        )
        if self.num_snakes == 1:
            return self.state.observe()
        else:
            return dict(
                zip(
                    map(str, range(self.num_snakes)),
                    [self.state.observe(i) for i in range(self.num_snakes)],
                )
            )

    def step(self, action: Union[Dict[str, int], int]):

        data = self.state.move_snakes(action)

        rewards, terminals = self._evaluate_reward(data)

        if self.num_snakes == 1:
            next_state = self.state.observe()
            reward = rewards[0]
            terminal = terminals[0]
        else:
            next_state = dict(
                zip(action.keys(), [self.state.observe(int(i)) for i in action.keys()])
            )
            reward = dict(zip(action.keys(), rewards))
            terminal = dict(zip(action.keys(), terminals))
            terminal["__all__"] = (
                self.num_snakes - len([s for s in self.state.snakes if s.is_dead()])
                <= 1
            )

        return next_state, reward, terminal, {}

    def render(self):
        return self.state.observe()

    def _evaluate_reward(self, data):
        rewards = []
        terminals = []
        for fruit_eaten, collided, starved, won in zip(*data):
            terminal = False
            reward = Reward.nothing.value
            if collided:
                terminal = True
                reward = Reward.collision.value
            else:
                if won:
                    reward = Reward.won.value
                    terminal = True
                elif fruit_eaten:
                    reward = Reward.fruit.value
                elif starved:
                    terminal = True
                    reward = Reward.starve.value
            rewards.append(reward)
            terminals.append(terminal)
        return rewards, terminals
