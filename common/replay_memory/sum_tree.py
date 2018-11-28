import logging

import numpy as np

from common.models.transition import Transition

LOGGER = logging.getLogger("PrioritizedBuffer")


class SumTree:

    """
    Stores the transitions in a sum tree that is represented in BFS order in an array.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.write_pointer = 0
        self.size = 0
        # Binary sum tree of priorities
        self.tree = np.zeros(2 * capacity - 1)
        self.transitions = np.zeros(capacity, dtype=np.ndarray)
        self.priorities = np.zeros(capacity)
        self.sampling_counter = np.zeros(capacity, dtype=int)

    def total(self):
        return self.tree[0]

    def add(self, priority: float, transition: Transition):
        if self.size < self.capacity:
            self.size += 1
        idx = self.write_pointer + self.capacity - 1

        self.transitions[self.write_pointer] = transition
        self.priorities[self.write_pointer] = priority
        self.sampling_counter[self.write_pointer] = 0
        self.update(idx, priority)

        transition_idx = self.write_pointer

        self.write_pointer += 1
        # If capacity is reached, reset write pointer
        if self.write_pointer >= self.capacity:
            LOGGER.info(f"Reached buffer maximum capacity.")
            self.write_pointer = 0
        return transition_idx

    def update(self, idx: int, priority: float):
        priority_difference = priority - self.tree[idx]
        self.priorities[idx - self.capacity + 1] = priority

        self.tree[idx] = priority
        self._propagate(idx, priority_difference)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        self.sampling_counter[dataIdx] += 1

        return (idx, self.tree[idx], self.transitions[dataIdx])

    def _propagate(self, idx: int, priority_difference: float):
        parent = int((idx - 1) / 2)

        self.tree[parent] += priority_difference

        if parent != 0:
            self._propagate(parent, priority_difference)

    def _retrieve(self, idx: int, s: float):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
