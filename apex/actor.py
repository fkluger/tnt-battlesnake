import atexit
import logging
import pickle
import math
import random
import zlib
import zmq

import numpy as np

from dqn.network import DQN
from apex.models import Experience, Observation
from apex.utils import get_ip_address
from .actor_statistics import ActorStatistics

LOGGER = logging.getLogger('Actor')


class Actor:

    def __init__(self, config, actor_idx, tensorboard_logger):
        self.stats = ActorStatistics(config, actor_idx, tensorboard_logger)
        self.buffer = list()
        self.episode_buffer = list()
        self.config = config
        self.idx = actor_idx
        self.epsilon = np.power(self.config.epsilon_base, (self.idx / self.config.get_num_actors()) * 7)
        LOGGER.info(f'Epsilon: {self.epsilon}')
        self.input_shape = (config.width, config.height, config.stacked_frames)
        self.dqn = DQN(input_shape=self.input_shape, num_actions=3, learning_rate=config.learning_rate)
        learner_address = config.learner_ip_address + ':' + config.starting_port
        self._connect_sockets(learner_address)

    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(3)
        else:
            q_values = self.dqn.predict(state)
            best_action = np.argmax(q_values)
            return best_action

    def observe(self, observation):
        self.episode_buffer.append(observation)
        self.stats.on_observe(observation)

        if observation.next_state is None:
            self.buffer += self._compute_multistep_bootstrap(self.episode_buffer)
            self.episode_buffer.clear()
            if len(self.buffer) >= self.config.actor_buffer_size:
                self.send_experiences()
                self.buffer.clear()

    def _decompress_weights(self, weights):
        return pickle.loads(weights)

    def update_parameters(self):
        try:
            message = self.parameter_socket.recv_multipart(flags=zmq.NOBLOCK)
            online_weights_pickled, target_weights_pickled = message[1], message[2]
            online_weights, target_weights = self._decompress_weights(
                online_weights_pickled), self._decompress_weights(target_weights_pickled)
            self.dqn.online_model.set_weights(online_weights)
            self.dqn.target_model.set_weights(target_weights)
            LOGGER.info('Received parameter update from learner.')
            return True
        except zmq.Again:
            return False

    def send_experiences(self):
        _, _, errors = self.dqn.create_targets(self.buffer, len(self.buffer))
        experiences = [Experience(observation, errors[idx]) for idx, observation in enumerate(self.buffer)]
        experiences_pickled = pickle.dumps(experiences, -1)
        experiences_compressed = zlib.compress(experiences_pickled)
        self.experience_socket.send_multipart([b'experiences', experiences_compressed])

    def _compute_multistep_bootstrap(self, episode_observations):
        episode_length = len(episode_observations)
        for idx, obs in enumerate(episode_observations):
            multi_step_reward = obs.reward
            nth_observation = obs
            n = 0
            for i in range(1, self.config.multi_step_n):
                if idx + i < episode_length:
                    multi_step_reward += np.power(self.config.discount_factor, i) * episode_observations[idx + i].reward
                    nth_observation = episode_observations[idx + i]
                    n = i
                else:
                    break
            obs.reward = multi_step_reward
            obs.discount_factor = np.power(self.config.discount_factor, np.amin([self.config.multi_step_n, n + 1]))
            obs.next_state = nth_observation.next_state
        return episode_observations

    def _connect_sockets(self, learner_address):
        self.context = zmq.Context()
        self.parameter_socket = self.context.socket(zmq.SUB)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.connect(f'tcp://{learner_address}')
        self.parameter_socket.setsockopt(zmq.SUBSCRIBE, b'parameters')
        LOGGER.info(f'Connected socket to learner at {learner_address}')

        self.experience_socket = self.context.socket(zmq.PUB)
        self.experience_socket.setsockopt(zmq.LINGER, 0)
        ip_address = get_ip_address()
        port = int(self.config.starting_port) + self.idx
        self.experience_socket.bind(f'tcp://{ip_address}:{port}')
        LOGGER.info(f'Created socket at {ip_address}:{port}')
        atexit.register(self._disconnect_sockets)

    def _disconnect_sockets(self):
        self.experience_socket.close()
        self.parameter_socket.close()
        self.context.term()
