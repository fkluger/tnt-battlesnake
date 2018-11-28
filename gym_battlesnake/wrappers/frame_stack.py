import gym
import numpy as np

from gym_battlesnake.envs import BattlesnakeEnv
from gym_battlesnake.envs.state import State


class FrameStack(gym.Wrapper):
    def __init__(self, env: BattlesnakeEnv, num_stacked_frames: int):
        super().__init__(env)
        self.num_stacked_frames = num_stacked_frames
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_stacked_frames, env.width, env.height),
            dtype=np.uint8,
        )

    def reset(self):
        self.unwrapped.state = State(
            width=self.unwrapped.width,
            height=self.unwrapped.height,
            num_snakes=self.unwrapped.num_snakes,
            num_fruits=self.unwrapped.num_fruits,
            stacked_frames=self.num_stacked_frames,
        )
        return self.unwrapped.state.observe()

    def step(self, action: int):
        return self.env.step(action)
