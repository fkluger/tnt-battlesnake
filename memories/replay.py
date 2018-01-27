import numpy as np
from . import Memory


class ReplayMemory(Memory):
    '''
    Memory that samples observations with uniform probability.
    '''

    observations = []

    def __init__(self, capacity=1000000):
        self.capacity = capacity

    def size(self):
        return len(self.observations)

    def add(self, observation):
        self.observations.insert(0, observation)
        if len(self.observations) > self.capacity:
            self.observations.pop()

    def sample(self, n, beta):
        n = min(n, len(self.observations))
        indices = np.random.randint(0, len(self.observations) - 1, n, dtype=int)
        batch = [o for i, o in enumerate(self.observations) if i in indices]
        return batch, indices

    def update(self, idx, error):
        pass
