from typing import List

import numpy as np

from common.models.transition import Transition
from common.models.agent import Agent


class RandomAgent(Agent):
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def act(self, state: np.ndarray):
        return np.random.choice(self.num_actions)

    def observe(self, transitions: List[Transition]):
        pass
