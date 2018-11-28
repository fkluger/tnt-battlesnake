from typing import List

import gym
from gym_battlesnake.envs import BattlesnakeEnv

from .multiplayer_agent import MultiplayerAgent


class Multiplayer(gym.Wrapper):
    def __init__(self, env: BattlesnakeEnv, enemy_agents: List[MultiplayerAgent]):
        super().__init__(env)
        env.num_snakes = len(enemy_agents) + 1
        self.enemy_agents = enemy_agents

    def step(self, action: int):
        actions = [action]

        for idx, enemy in enumerate(self.enemy_agents):
            # Index 0 is the agent snake
            enemy_action = enemy.act(self.unwrapped.state, idx + 1)
            actions.append(enemy_action)

        return self.env.step(actions)

    def reset(self):
        return self.unwrapped.reset()
