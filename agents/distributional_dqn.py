import random
import math
import numpy as np
from collections import deque
from scipy.stats import kendalltau

from brains.huber_loss import create_np_quantile_huber_loss
from .dqn import DQNAgent


class DistributionalDQNAgent(DQNAgent):

    q_value_quantile_history = deque([], maxlen=10000)
    kendall_tau_history = []

    def __init__(self, num_quantiles, **kwargs):
        self.num_quantiles = num_quantiles
        self.optimal_quantile_ordering = np.arange(self.num_quantiles - 1, -1, -1)
        self.huber_loss = create_np_quantile_huber_loss(self.num_quantiles)
        super().__init__(**kwargs)

    def get_metrics(self):
        metrics = [
            # {'name': 'epsilon', 'value': self.epsilon, 'type': 'value'}, 
            {'name': 'quantile_histogram', 'value': self.q_value_quantile_history, 'type': 'histogram'},
            {'name': 'mean quantile ordering coefficient', 'value': np.mean(self.kendall_tau_history), 'type': 'value'}
        ]
        self.kendall_tau_history = []
        return metrics

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            quantiles = np.array(self.brain.predict(np.expand_dims(state, 0)))
            quantiles = np.swapaxes(quantiles, 0, 1)
            best_action = self.compute_best_action(quantiles)
            return best_action

    def compute_best_action(self, quantiles):
        quantiles_mean = np.mean(quantiles, axis=2)
        return np.argmax(quantiles_mean, axis=1)

    def observe(self, observation):
        state, action, reward, next_state = observation[0], observation[1], observation[2], observation[3]
        q_value_quantiles = np.array(self.brain.predict(np.expand_dims(state, 0)))
        q_value_quantiles = np.swapaxes(q_value_quantiles, 0, 1)
        if next_state is None:
            next_state = np.zeros(self.input_shape)
        q_value_quantiles_next = np.array(self.brain.predict(np.expand_dims(next_state, 0)))
        q_value_quantiles_next = np.swapaxes(q_value_quantiles_next, 0, 1)

        best_action = self.compute_best_action(q_value_quantiles)

        target = q_value_quantiles[0, :, :]
        target_old = list(target[action])
        reward = np.tile([reward], self.num_quantiles)
        if next_state is None:
            target[action] = reward
        else:
            target[action] = reward + self.GAMMA * q_value_quantiles_next[0, best_action[0], :]
        
        error = self.huber_loss(target_old, target[action])
        
        self.memory.add(observation, error)
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.brain.update_target()
        self.epsilon = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * math.exp(-self.LAMBDA * self.steps)
        self.beta += (1. - self.beta) / 1e6

    def replay(self):
        batch, indices, weights = self.memory.sample(self.batch_size, self.beta)

        # Actual batch size can differ from self.batch_size if the memory is not filled yet
        batch_size = len(batch)

        no_state = np.zeros(self.input_shape)
        next_states = np.array([(no_state if observation[3] is None else observation[3])
                                for observation in batch])
        states = np.array([observation[0] for observation in batch])

        q_value_quantiles = np.array(self.brain.predict(states))  # shape: (num_actions, batch_size, num_quantiles)
        q_value_quantiles = np.swapaxes(q_value_quantiles, 0, 1)  # shape: (batch_size, num_actions, num_quantiles)

        # shape: (num_actions, batch_size, num_quantiles)
        q_value_quantiles_next = np.array(self.brain.predict(next_states, target=True))
        # shape: (batch_size, num_actions, num_quantiles)
        q_value_quantiles_next = np.swapaxes(q_value_quantiles_next, 0, 1)

        best_action = self.compute_best_action(q_value_quantiles)

        x = np.zeros((batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.zeros((batch_size, self.num_actions, self.num_quantiles))
        errors = np.zeros(batch_size)
        for i, observation in enumerate(batch):
            state, action, reward, next_state = observation[0], observation[1], observation[2], observation[3]

            target = q_value_quantiles[i, :, :]
            target_old = list(target[action])
            reward = np.tile([reward], self.num_quantiles)
            if next_state is None:
                target[action] = reward
            else:
                target[action] = reward + self.GAMMA * q_value_quantiles_next[i, best_action[i], :]
            x[i] = state
            y[i] = target
            errors[i] = self.huber_loss(target_old, target[action])
            self.q_value_quantile_history.append(np.squeeze(target[action]))
            self.kendall_tau_history.append(kendalltau(np.argsort(np.squeeze(target[action])), self.optimal_quantile_ordering))
            self.memory.update(indices[i], errors[i])

        y = [y[:, i, :] for i in range(self.num_actions)]
        weights = [weights, weights, weights]
        history = self.brain.train(x, y, batch_size, weights)
        loss = history.history['loss'][0]
        return loss, 0
