import random
import math
import numpy as np

from . import Agent
from brains.huber_loss import np_huber_loss


class DQNAgent(Agent):
    '''
    DQN Agent that uses epsilon-greedy for exploration.
    '''

    steps = 0
    epsilon = 0

    def __init__(self, brain, memory, input_shape, num_actions, GAMMA=0.9, EPSILON_MAX=1, EPSILON_MIN=0.1, LAMBDA=1e-4, batch_size=32, update_target_freq=10000, replay_beta_min=0.4, multi_step_n=10):
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
        self.update_target_freq = update_target_freq
        self.beta = replay_beta_min
        self.multi_step_n = multi_step_n

    def get_metrics(self):
        return [{'name': 'epsilon', 'value': self.epsilon, 'type': 'value'}]

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.brain.predict(state[np.newaxis, ...]))

    def observe(self, observation):
        state, next_state = observation[0], observation[3]
        q_values = self.brain.predict(state[np.newaxis, ...])
        if next_state is None:
            next_state = np.zeros(self.input_shape)
        q_values_next = self.brain.predict(next_state[np.newaxis, ...])

        _, _, error = self.create_target(observation, q_values[0], q_values_next[0])
        self.memory.add(observation, error)
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.brain.update_target()
        self.epsilon = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * math.exp(-self.LAMBDA * self.steps)
        self.beta += (1. - self.beta) / 1e6
    
    def create_target(self, observation, q_values, q_values_next):
        state, action, reward, next_state = observation[0], observation[1], observation[2], observation[3]

        target = q_values
        target_old = target[action]
        if next_state is None:
            target[action] = reward
        else:
            target[action] = reward + self.GAMMA * np.amax(q_values_next)
        
        error = np_huber_loss(target[action], target_old)
        
        return state, target, error
        

    def replay(self):
        batch, indices, weights = self.memory.sample(self.batch_size, self.beta)

        no_state = np.zeros(self.input_shape)
        next_states = np.array([(no_state if observation[3] is None else observation[3])
                                for observation in batch])
        states = np.array([observation[0] for observation in batch])

        q_values = self.brain.predict(states)
        q_values_next = self.brain.predict(next_states, target=True)

        x = np.zeros((len(batch), self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.zeros((len(batch), self.num_actions))
        errors = np.zeros(len(batch))
        q_value_estimates = np.zeros(len(batch))
        for i, observation in enumerate(batch):
            state, target, error = self.create_target(observation, q_values[i], q_values_next[i])

            x[i] = state
            y[i] = target
            errors[i] = error
            q_value_estimates[i] = target[observation[1]]
            self.memory.update(indices[i], errors[i])

        history = self.brain.train(x, y, len(batch), weights)
        loss = history.history['loss'][0]
        return loss, np.mean(q_value_estimates)
