from abc import ABC, abstractmethod
from typing import List


class ExplorationStrategy(ABC):

    """
    Strategy in Reinforcement Learning to balance the exploration/exploitation trade-off.
    """

    @abstractmethod
    def choose_action(self, q_values: List[float], episode: int) -> int:
        pass
