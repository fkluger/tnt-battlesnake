import random
from . import Memory
from .sum_tree import SumTree


class PrioritizedReplayMemory(Memory):
    '''
    Memory that samples observations proportional to their time-difference error.
    '''

    e = 0.01
    a = 0.6
    initial_priority = 1

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def size(self):
        return self.tree.size

    def add(self, observation):
        self.tree.add(self.initial_priority, observation)

    def sample(self, n):
        batch = []
        indices = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, observation) = self.tree.get(s)
            batch.append(observation)
            indices.append(idx)

        return batch, indices

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    def _getPriority(self, error):
        return (error + self.e) ** self.a
