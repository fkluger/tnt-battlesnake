import random
import numpy as np
from . import Memory
from .sum_tree import SumTree


class PrioritizedReplayMemory(Memory):
    '''
    Memory that samples observations proportional to their time-difference error.
    '''

    def __init__(self, capacity, epsilon, alpha, max_priority):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_priority = max_priority

    def size(self):
        return self.tree.size

    def add(self, observation):
        self.tree.add(self.max_priority, observation)

    def sample(self, n, beta):
        batch = []
        indices = []
        weights = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, priority, observation) = self.tree.get(s)
            weights.append((self.capacity * priority)**-beta)
            batch.append(observation)
            indices.append(idx)

        weights /= max(weights)
        return batch, indices, weights

    def update(self, idx, error):
        p = self._getPriority(error)
        self.max_priority = max(p, self.max_priority)
        self.tree.update(idx, p)

    def _getPriority(self, error):
        return (error + self.epsilon) ** self.alpha
