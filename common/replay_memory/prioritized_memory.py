from typing import List, Tuple

import numpy as np

from common.models.transition import Transition

from .sum_tree import SumTree


class PrioritizedMemory:
    """
    Buffer that samples transitions proportional to their time-difference error.
    """

    def __init__(
        self, capacity: int, epsilon: float, alpha: float, max_priority: float
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_priority = max_priority

    def size(self):
        return self.tree.size

    def add(self, transition: Transition, error=None):
        if error is None:
            priority = self.max_priority
        else:
            priority = self._getPriority(error)
        return self.tree.add(priority, transition)

    def sample(
        self, batch_size: int, beta: float
    ) -> Tuple[List[Transition], List[int], List[float]]:
        transitions = np.ndarray(batch_size, dtype=tuple)
        indices = np.ndarray(batch_size, dtype=int)
        weights = np.ndarray(batch_size, dtype=float)

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            (idx, priority, transition) = self.tree.get(s)

            # importance sampling weight
            weight = np.power(self.capacity * priority, -beta)

            weights[i] = weight
            indices[i] = idx
            transitions[i] = transition

        # Normalize weights to stabilize updates
        weights /= max(weights)

        return transitions, indices, weights

    def update(self, indices: np.ndarray, errors: np.ndarray):
        for idx, error in zip(indices, errors):
            self._update(idx, error)

    def _update(self, idx: int, error: float):
        p = self._getPriority(error)
        self.max_priority = max(p, self.max_priority)
        self.tree.update(idx, p)

    def _getPriority(self, error: float):
        return (error + self.epsilon) ** self.alpha
