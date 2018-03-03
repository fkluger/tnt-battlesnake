import random
import math
import numpy as np
from collections import deque

from .dqn import DQNAgent
from brains.huber_loss import create_np_quantile_huber_loss

class DistributionalDQNAgent(DQNAgent):

    q_value_quantile_history = deque([], maxlen=10000)
    quantile_ordering_error_history = []
    episode_observations = []
    episode_observation_indices = []

    def __init__(self, num_quantiles, **kwargs):
        self.num_quantiles = num_quantiles
        self.quantile_huber_loss = create_np_quantile_huber_loss(self.num_quantiles)
        self.optimal_quantile_ordering = np.arange(self.num_quantiles)
        super().__init__(**kwargs)

    def get_metrics(self):
        metrics = [
            {'name': 'agent/quantile_histogram', 'value': self.q_value_quantile_history, 'type': 'histogram'},
            {'name': 'agent/mean quantile ordering error', 'value': np.mean(self.quantile_ordering_error_history), 'type': 'value'}
        ]
        self.quantile_ordering_error_history = []
        return metrics

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            quantiles = np.array(self.brain.predict(np.expand_dims(state, 0)))
            best_action = self.compute_best_action(quantiles)
            return np.squeeze(best_action)

    def compute_best_action(self, quantiles):
        quantiles_mean = np.mean(quantiles, axis=-1)
        return np.argmax(quantiles_mean, axis=-2)
    
    def update_episode_observations(self):
        for idx, observation in enumerate(self.episode_observations):
            n_step_reward = observation[2]
            next_state = observation[3]
            for t in range(1, self.multi_step_n):
                next_t = idx + t
                if next_t >= len(self.episode_observations):
                    next_state = None
                    break
                else:
                    n_step_reward += (self.GAMMA**t) * self.episode_observations[next_t][2]
                    next_state = self.episode_observations[next_t][0]
            self.memory.update_observation(self.episode_observation_indices[idx], (observation[0], observation[1], n_step_reward, next_state))
        self.episode_observation_indices = []
        self.episode_observations = []

    def observe(self, observation):
        self.episode_observations.append(observation)
        self.episode_observation_indices.append(self.memory.add(observation))
        if observation[3] is None:
            self.update_episode_observations()
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.brain.update_target()
        self.epsilon = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * math.exp(-self.LAMBDA * self.steps)
        self.beta += (1. - self.beta) * self.LAMBDA

    def replay(self):
        batch, indices, weights = self.memory.sample(self.batch_size, self.beta)

        # Actual batch size can differ from self.batch_size if the memory is not filled yet
        batch_size = len(batch)

        no_state = np.zeros(self.input_shape)
        next_states = np.array([(no_state if observation[3] is None else observation[3])
                                for observation in batch])
        states = np.array([observation[0] for observation in batch])

        q_value_quantiles = np.array(self.brain.predict(states))
        q_value_quantiles_next = np.array(self.brain.predict(next_states, target=True))
        # shape: (batch_size, num_actions, num_quantiles)

        best_action = self.compute_best_action(q_value_quantiles)

        x = np.zeros((batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.zeros((batch_size, self.num_actions, self.num_quantiles))
        errors = np.zeros(batch_size)
        for i, observation in enumerate(batch):
            state, action, reward, next_state = observation[0], observation[1], observation[2], observation[3]

            target = q_value_quantiles[i]
            target_old = np.copy(target)
            reward = np.tile([reward], self.num_quantiles)
            if next_state is None:
                target[action] = reward
            else:
                target[action] = reward + (self.GAMMA**self.multi_step_n) * q_value_quantiles_next[i, best_action[i]]
            x[i] = state
            y[i] = target
            errors[i] = self.quantile_huber_loss(np.expand_dims(target, 0), np.expand_dims(target_old, 0))
            self.q_value_quantile_history.append(np.squeeze(target[action]))
            self.quantile_ordering_error_history.append(np.linalg.norm(np.argsort(np.squeeze(target_old[action])) - self.optimal_quantile_ordering))
            self.memory.update(indices[i], errors[i])
        self.brain.train(x, y, batch_size, weights)
        
