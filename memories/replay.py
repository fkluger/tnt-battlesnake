import random
from . import Memory


class ReplayMemory(Memory):
    '''
    Memory that samples observations with uniform probability.
    '''

    observations = []

    def __init__(self, capacity=100000):
        self.capacity = capacity

    def add(self, observation):
        self.observations.insert(0, observation)
        if len(self.observations) > self.capacity:
            self.observations.pop()

    def sample(self, n):
        n = min(n, len(self.observations))
        return random.sample(self.observations, n)
