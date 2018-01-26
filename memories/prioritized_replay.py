import random
from . import Memory
from .sum_tree import SumTree


class PrioritizedReplayMemory(Memory):
    '''
    Memory that samples observations proportional to their time-difference error.
    '''

    epsilon = 0.01
    alpha = 0.9
    max_priority = 1

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def size(self):
        return self.tree.size

    def add(self, observation):
        self.tree.add(self.max_priority, observation)

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
        self.max_priority = max(p, self.max_priority)
        self.tree.update(idx, p)

    def _getPriority(self, error):
        return (error + self.epsilon) ** self.alpha
