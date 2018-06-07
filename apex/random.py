import numpy as np

from . import Agent


class RandomAgent(Agent):
    '''
    Random agent.
    '''

    steps = 0

    def __init__(self, memory, num_actions):
        self.memory = memory
        self.num_actions = num_actions

    def get_metrics(self):
        return []

    def act(self, state):
        return np.random.randint(0, self.num_actions - 1)

    def observe(self, observation):
        self.memory.add(observation)
        self.steps += 1

    def replay(self):
        pass
