import random
import math
import numpy as np
from . import Agent


class DQNAgent(Agent):

    steps = 0
    epsilon = 0

    def __init__(self, brain, memory, input_shape, num_actions, GAMMA=0.99, EPSILON_MAX=1, EPSILON_MIN=0.01, LAMBDA=0.001, batch_size=64):
        self.brain = brain
        self.memory = memory
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON_MIN = EPSILON_MIN
        self.LAMBDA = LAMBDA
        self.GAMMA = GAMMA
        self.epsilon = EPSILON_MAX
        self.batch_size = batch_size

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.brain.predict(state[np.newaxis, ...]))

    def observe(self, sample):
        self.memory.add(sample)
        self.steps += 1
        self.epsilon = self.EPSILON_MIN + \
            (self.EPSILON_MAX - self.EPSILON_MIN) * \
            math.exp(-self.LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(self.batch_size)

        no_state = np.zeros(self.input_shape)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3])
                            for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = np.zeros((len(batch), self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.zeros((len(batch), self.num_actions))
        for i, o in enumerate(batch):
            state, action, reward, next_state = o[0], o[1], o[2], o[3]

            target = p[i]
            if next_state is None:
                target[action] = reward
            else:
                target[action] = reward + self.GAMMA * np.amax(p_[i])

            x[i] = state
            y[i] = target

        self.brain.train(x, y, self.batch_size)
