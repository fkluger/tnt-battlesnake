from abc import ABC, abstractmethod

import numpy as np

from .transition import Transition


class Agent(ABC):
    @abstractmethod
    def act(self, state: np.ndarray):
        pass

    @abstractmethod
    def observe(self, transition: Transition):
        pass
