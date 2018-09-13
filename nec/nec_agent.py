from typing import List
import tensorflow as tf
import numpy as np

from apex import Observation, Configuration
from replay_buffer import PrioritizedBuffer

from .dnd import DifferentiableNeuralDictionary
from .build_graph import build_graph
from .encoder import create_encoder


class NECAgent:
    def __init__(self, config: Configuration):
        self.config = config
        self.beta = config.replay_importance_weight
        self.epsilon = config.epsilon_base
        self.episode_buffer = []
        self.write_buffer = []
        self.replay_buffer = PrioritizedBuffer(
            capacity=config.replay_capacity,
            epsilon=config.replay_min_priority,
            alpha=config.replay_prioritization_factor,
            max_priority=config.replay_max_priority,
        )
        self.dnds = [
            DifferentiableNeuralDictionary(
                a,
                config.nec_capacity,
                config.nec_key_length,
                config.nec_num_nearest_neighbours,
                config.nec_delta,
                config.nec_learning_rate,
            )
            for a in range(config.num_actions)
        ]
        self.encoder = create_encoder(config.get_input_shape(), config.nec_key_length)
        self._train, self._act, self._write = build_graph(
            self.encoder,
            tf.train.AdamOptimizer(config.learning_rate),
            self.dnds,
            config.get_input_shape(),
            config.num_actions,
            config.nec_key_length,
        )

    def act(self, state):
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            return np.random.choice(self.config.num_actions), False
        else:
            if len(state.shape) == 3:
                state = np.expand_dims(state, 0)
            best_action, _ = self._act(state)
            return best_action, True

    def observe(self, observation: Observation):
        # TODO: Compute error
        self.replay_buffer.add(observation)
        self.episode_buffer.append(observation)
        if observation.next_state is None:
            self.episode_buffer = self._compute_multistep_bootstrap(self.episode_buffer)
            self.write_buffer.extend(self.episode_buffer)
            self.episode_buffer.clear()

            if len(self.write_buffer) > 1000:
                observations_per_action = [
                    [obs for obs in self.episode_buffer if obs.action == a]
                    for a in range(self.config.num_actions)
                ]
                states_per_action = [
                    [obs.state for obs in self.episode_buffer if obs.action == a]
                    for a in range(self.config.num_actions)
                ]
                q_values_per_action = [
                    self._compute_q_values(observations)
                    for observations in observations_per_action
                ]
                self._write(states_per_action, q_values_per_action)
                self.write_buffer.clear()

    def train(self):
        self._update_importance_weight_coefficient()
        batch, indices, weights = self.replay_buffer.sample(
            self.config.batch_size, self.beta
        )
        states = [obs.state for obs in batch]
        actions = [obs.action for obs in batch]
        target_q_values = self._compute_q_values(batch)

        error = self._train(states, actions, target_q_values)
        print(f"Error: {error}")

    def _compute_q_values(self, observations: List[Observation]):
        q_values = np.zeros(len(observations))
        if len(observations):
            _, q_values_next = self._act(
                [
                    (
                        obs.next_state
                        if obs.next_state is not None
                        else np.zeros(self.config.get_input_shape())
                    )
                    for obs in observations
                ]
            )
            for idx, obs in enumerate(observations):
                if obs.next_state is None:
                    q_values[idx] = obs.reward
                else:
                    q_values[idx] = obs.reward + obs.discount_factor * np.amax(
                        np.transpose(q_values_next)[idx]
                    )
        return q_values

    def _compute_multistep_bootstrap(self, episode_observations: List[Observation]):
        episode_length = len(episode_observations)
        for idx, obs in enumerate(episode_observations):
            multi_step_reward = obs.reward
            nth_observation = obs
            n = 0
            for i in range(1, self.config.multi_step_n):
                if idx + i < episode_length and nth_observation.greedy:
                    multi_step_reward += (
                        np.power(self.config.discount_factor, i)
                        * episode_observations[idx + i].reward
                    )
                    nth_observation = episode_observations[idx + i]
                    n = i
                else:
                    break
            obs.reward = multi_step_reward
            obs.discount_factor = np.power(
                self.config.discount_factor, np.amin([self.config.multi_step_n, n + 1])
            )
            obs.next_state = nth_observation.next_state
        return episode_observations

    def _update_importance_weight_coefficient(self):
        self.beta += (
            1. - self.beta
        ) * self.config.replay_importance_weight_annealing_step_size
