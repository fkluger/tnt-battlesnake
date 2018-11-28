from abc import ABC, abstractmethod

from gym_battlesnake.envs.state import State


class MultiplayerAgent(ABC):
    @abstractmethod
    def act(self, state: State, index: int):
        pass
