from abc import ABC, abstractmethod


class Simulator(ABC):

    @abstractmethod
    def reset(self):
        return

    @abstractmethod
    def step(self, actions):
        return
