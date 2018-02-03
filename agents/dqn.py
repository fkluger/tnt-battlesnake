import random
import math
import numpy as np

from . import Agent


class DQNAgent(Agent):
    '''
    DQN Agent that uses epsilon-greedy for exploration.
    '''

    steps = 0
    epsilon = 0
    episode_observations = []
    episode_observation_indices = []

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
        self.episode_observations.append(observation)
        self.episode_observation_indices.append(self.memory.add(observation))
        if observation[3] is None:
            self.update_episode_observations()
            self.episode_observation_indices = []
            self.episode_observations = []
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.brain.update_target()
        self.epsilon = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * math.exp(-self.LAMBDA * self.steps)
        self.beta += (1. - self.beta) / 1e6
    
    def update_episode_observations(self):
        for idx, observation in enumerate(self.episode_observations):
            n_step_reward = observation[2]
            for t in range(1, self.multi_step_n + 1):
                next_t = idx + t
                next_state = None
                if next_t >= len(self.episode_observations):
                    break
                else:
                    n_step_reward += (self.GAMMA**(t - 1)) * self.episode_observations[next_t][2]
                    next_state = self.episode_observations[next_t][0]
            self.memory.update_observation(self.episode_observation_indices[idx], (observation[0], observation[1], n_step_reward, next_state))

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
            state, action, reward, next_state = observation[0], observation[1], observation[2], observation[3]

            target = q_values[i]
            target_old = target[action]
            if next_state is None:
                target[action] = reward
            else:
                target[action] = reward + self.GAMMA * np.amax(q_values_next[i])

            x[i] = state
            y[i] = target
            errors[i] = abs(target_old - target[action])
            q_value_estimates[i] = target[action]
            self.memory.update(indices[i], errors[i])

        history = self.brain.train(x, y, len(batch), weights)
        loss = history.history['loss'][0]
        return loss, np.mean(q_value_estimates)
