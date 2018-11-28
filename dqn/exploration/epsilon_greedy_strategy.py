from typing import List

import numpy as np

from .exploration_strategy import ExplorationStrategy


class EpsilonGreedyStrategy(ExplorationStrategy):

    """
    Exploration strategy for Reinforcement Learning that allows to learn the optimal policy.
    """

    def __init__(self, epsilon_max: float, epsilon_min: float, epsilon_decay: float):
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.last_episode = 0

    def choose_action(self, q_values: List[float], episode: int) -> int:
        """
        Choose an action using the Q-values.

        Arguments:
            q_values {`List[float]`} -- Computed Q-values

        Returns:
            `int` -- With probability epsilon a random action, else the argmax of the Q-values
        """
        if episode != self.last_episode:
            self._update_epsilon()
            self.last_episode = episode
        if np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)

    def _update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
