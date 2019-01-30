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
            shape=(
                env.observation_space.shape[1],
                env.observation_space.shape[2],
                num_stacked_frames,
            ),
            dtype=np.uint8,
        )

    def reset(self):
        self.unwrapped.state = State(
            width=self.unwrapped.width,
            height=self.unwrapped.height,
            num_snakes=self.unwrapped.num_snakes,
            num_fruits=self.unwrapped.num_fruits,
            stacked_frames=self.num_stacked_frames,
            window_width=self.unwrapped.window_width,
            window_height=self.unwrapped.window_height,
        )
        obs = self.unwrapped.state.observe()
        obs = np.moveaxis(obs, 0, -1)
        return obs

    def step(self, action: int):
        obs, rew, done, info = self.env.step(action)
        if obs is None:
            obs = np.zeros(
                shape=(self.observation_space.shape), dtype=self.observation_space.dtype
            )
        else:
            obs = np.moveaxis(obs, 0, -1)
        return obs, rew, done, info
