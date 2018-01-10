import random
from . import Memory

class ReplayMemory(Memory):
    
    samples = []

    def __init__(self, capacity=100000):
        self.capacity = capacity

    def add(self, sample):
        self.samples.insert(0, sample)
        if len(self.samples) > self.capacity:
            self.samples.pop()

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
