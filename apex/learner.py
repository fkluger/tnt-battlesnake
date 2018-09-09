import atexit
import logging
import pickle
import os
import zlib
import zmq

from dqn.network import DQN
from replay_buffer.prioritized_buffer import PrioritizedBuffer
from tensorboard_logger import TensorboardLogger
from apex.configuration import Configuration

from .learner_statistics import LearnerStatistics

LOGGER = logging.getLogger("Learner")


class Learner:
    def __init__(self, config: Configuration):
        self.config = config
        self.tensorboard_logger = TensorboardLogger(self.config.output_directory)
        self.input_shape = (config.width, config.height, self.config.stacked_frames)
        self.dqn = DQN(
            input_shape=self.input_shape,
            num_actions=3,
            learning_rate=config.learning_rate,
        )

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
        self.target_weights_changed = False
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

    def update_experiences(self):
        try:
            message = self.experiences_socket.recv_multipart(flags=zmq.NOBLOCK)
            experiences_compressed = message[1]
            experiences_pickled = zlib.decompress(experiences_compressed)
            experiences = pickle.loads(experiences_pickled)
            for experience in experiences:
                self.buffer.add(experience.observation, experience.error)
            self.beta += (
                1. - self.beta
            ) * self.config.replay_importance_weight_annealing_step_size
            self.stats.on_batch_receive(experiences)
            if self.stats.received_batches % self.config.training_interval == 0:
                self.evaluate_experiences()
            return True
        except zmq.Again:
            return False

    def evaluate_experiences(self):
        if self.buffer.size() <= self.config.batch_size:
            return
        if self.stats.training_batches % self.config.target_update_interval == 0:
            self.dqn.update_target_model()
            self.target_weights_changed = True
        batch, indices, weights = self.buffer.sample(self.config.batch_size, self.beta)
        # Actual batch size can differ from self.batch_size if the memory is not filled yet
        batch_size = len(batch)

        x, y, errors = self.dqn.create_targets(batch, batch_size)
        for idx in range(batch_size):
            self.buffer.update(indices[idx], errors[idx])
        loss = self.dqn.train(x, y, batch_size, weights)
        self.stats.on_evaluation(batch, errors, loss)

    def _compress_weights(self, weights):
        return pickle.dumps(weights, -1)

    def send_parameters(self):
        LOGGER.debug("Sending parameters...")
        online_weights = self.dqn.online_model.get_weights()
        self.stats.on_weight_export(self.dqn.online_model)
        online_weights_compressed = self._compress_weights(online_weights)
        if self.target_weights_changed:
            target_weights = self.dqn.target_model.get_weights()
            target_weights_compressed = self._compress_weights(target_weights)
            self.target_weights_changed = False
        else:
            target_weights_compressed = b"empty"
        self.parameter_socket.send_multipart(
            [b"parameters", online_weights_compressed, target_weights_compressed]
        )
