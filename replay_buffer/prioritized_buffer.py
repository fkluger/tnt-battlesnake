import numpy as np

from .sum_tree import SumTree

class PrioritizedBuffer:
    '''
    Buffer that samples observations proportional to their time-difference error.
    '''

    def __init__(self, capacity, epsilon, alpha, max_priority):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_priority = max_priority

    def size(self):
        return self.tree.size

    def add(self, observation, error=None):
        if error is None:
            priority = self.max_priority
        else:
            priority = self._getPriority(error)
        return self.tree.add(priority, observation)

    def sample(self, batch_size, beta):
        observations = np.ndarray(batch_size, dtype=tuple)
        indices = np.ndarray(batch_size, dtype=int)
        weights = np.ndarray(batch_size, dtype=float)

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            (idx, priority, observation) = self.tree.get(s)

            # importance sampling weight
            weight = np.power(self.capacity * priority, -beta)

            weights[i] = weight
            indices[i] = idx
            observations[i] = observation

        # Normalize weights to stabilize updates
        weights /= max(weights)

        return observations, indices, weights

    def update(self, idx, error):
        p = self._getPriority(error)
        self.max_priority = max(p, self.max_priority)
        self.tree.update(idx, p)

    def _getPriority(self, error):
        return (error + self.epsilon) ** self.alpha
