import atexit
import logging
import pickle
import os
import zlib
import zmq

from replay_buffer.prioritized_buffer import PrioritizedBuffer
from tensorboard_logger import TensorboardLogger

from .configuration import Configuration
from .learner_statistics import LearnerStatistics

LOGGER = logging.getLogger("Learner")


class AbstractLearner:
    def __init__(self, config: Configuration):
        self.config = config
        self.tensorboard_logger = TensorboardLogger(self.config.output_directory)
        self.buffer = PrioritizedBuffer(
            capacity=config.replay_capacity,
            epsilon=config.replay_min_priority,
            alpha=config.replay_prioritization_factor,
            max_priority=config.replay_max_priority,
        )
        self.beta = config.replay_importance_weight

        self.stats = LearnerStatistics(
            self.config, self.tensorboard_logger, self.buffer
        )
        learner_address = config.learner_ip_address + ":" + config.starting_port
        self._connect_sockets(learner_address)

    def _connect_sockets(self, learner_address):
        self.context = zmq.Context()
        self.parameter_socket = self.context.socket(zmq.PUB)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.bind(f"tcp://{learner_address}")
        LOGGER.info(f"Created socket at {learner_address}")

        self.experiences_socket = self.context.socket(zmq.SUB)
        self.experiences_socket.setsockopt(zmq.LINGER, 0)
        self.experiences_socket.setsockopt(zmq.SUBSCRIBE, b"experiences")
        for ip in self.config.actors.keys():
            for idx in range(self.config.actors[ip]):
                port = str(int(self.config.starting_port) + idx + 1)
                address = ip + ":" + port
                self.experiences_socket.connect(f"tcp://{address}")
                LOGGER.info(f"Connected socket to actor at {address}")
        atexit.register(self._disconnect_sockets)

    def _disconnect_sockets(self):
        self.parameter_socket.close()
        self.experiences_socket.close()
        self.context.term()

    def _update_experiences(self, experiences):
        pass

    def update_experiences(self):
        try:
            message = self.experiences_socket.recv_multipart(flags=zmq.NOBLOCK)
            experiences_compressed = message[1]
            experiences_pickled = zlib.decompress(experiences_compressed)
            experiences = pickle.loads(experiences_pickled)
            for experience in experiences:
                self.buffer.add(experience.observation, experience.error)
            self._update_experiences(experiences)
            self.beta += (
                1. - self.beta
            ) * self.config.replay_importance_weight_annealing_step_size
            self.stats.on_batch_receive(experiences)
            return True
        except zmq.Again:
            return False

    def _evaluate_experiences(self):
        pass

    def evaluate_experiences(self):
        if self.buffer.size() <= self.config.batch_size:
            return
        self._evaluate_experiences()

    def _create_parameters_message(self):
        pass

    def send_parameters(self):
        LOGGER.info("Sending parameters...")
        message = self._create_parameters_message()
        self.parameter_socket.send_multipart(message)
