import atexit
import logging
import pickle
import random
import zlib
import zmq

import numpy as np

from dqn.network import DQN
from apex.models import Experience
from apex.utils import get_free_port, get_ip_address

LOGGER = logging.getLogger('Actor')


class Actor:

    def __init__(self, config):
        self.received_parameter_updates = 0
        self.buffer = list()
        self.config = config
        self.input_shape = (config.width, config.height, 1)
        self.dqn = DQN(input_shape=self.input_shape, num_actions=3, learning_rate=config.learning_rate)
        learner_address = config.learner_ip_address + ':' + config.starting_port
        idx = self._connect_sockets(learner_address)

        self.epsilon = np.power(0.4, 1 + (idx / (self.config.get_num_actors() - 1)) * 7)

        LOGGER.info(f'Epsilon: {self.epsilon}')

    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(3)
        else:
            q_values = self.dqn.predict(state)
            best_action = np.argmax(q_values)
            LOGGER.debug(f'Q-Values: {q_values}, Action: {best_action}')
            return best_action

    def observe(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) >= self.config.actor_buffer_size:
            self.send_experiences()
            self.buffer.clear()

    def update_parameters(self):
        try:
            message = self.parameter_socket.recv_multipart(flags=zmq.NOBLOCK)
            online_weights_compressed, target_weights_compressed = message[1], message[2]
            online_weights_pickled, target_weights_pickled = zlib.decompress(online_weights_compressed), zlib.decompress(target_weights_compressed)
            online_weights, target_weights = pickle.loads(online_weights_pickled), pickle.loads(target_weights_pickled)
            self.received_parameter_updates += 1
            self.dqn.online_model.set_weights(online_weights)
            self.dqn.target_model.set_weights(target_weights)
            LOGGER.info('Received parameter update from learner.')
            return True
        except zmq.Again:
            return False

    def send_experiences(self):
        _, _, errors = self.dqn.create_targets(self.buffer, self.config.discount_factor, len(self.buffer))
        experiences = list()
        for idx, observation in enumerate(np.copy(self.buffer)):
            experiences.append(Experience(observation, errors[idx]))
        experiences_pickled = pickle.dumps(experiences, -1)
        experiences_compressed = zlib.compress(experiences_pickled)
        self.experience_socket.send_multipart([b'experiences', experiences_compressed])

    def _connect_sockets(self, learner_address):
        self.context = zmq.Context()
        self.parameter_socket = self.context.socket(zmq.SUB)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.connect(f'tcp://{learner_address}')
        self.parameter_socket.setsockopt(zmq.SUBSCRIBE, b'parameters')
        LOGGER.info(f'Connected socket to learner at {learner_address}')

        self.experience_socket = self.context.socket(zmq.PUB)
        self.experience_socket.setsockopt(zmq.LINGER, 0)
        port = get_free_port(int(self.config.starting_port) + 1)
        ip_address = get_ip_address()
        self.experience_socket.bind(f'tcp://{ip_address}:{port}')
        LOGGER.info(f'Created socket at {ip_address}:{port}')

        atexit.register(self._disconnect_sockets)
        return port - int(self.config.starting_port)

    def _disconnect_sockets(self):
        self.experience_socket.close()
        self.parameter_socket.close()
        self.context.term()
