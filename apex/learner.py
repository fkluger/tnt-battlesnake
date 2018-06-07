import atexit
import json
import pickle
import time
import zlib
import zmq
import numpy as np

from memories.prioritized_replay import PrioritizedReplayMemory
from simulator.utils import get_state_shape
from dqn import DQN
from utils import get_logger, create_targets
from tensorboard import CustomTensorboard

logger = get_logger('Learner')


class Learner:
    def __init__(self, config):
        self.config = config
        self.input_shape = get_state_shape(config['width'], config['height'],
                                           config['num_frames'])
        self.dqn = DQN(
            input_shape=self.input_shape,
            num_actions=3,
            hidden_size=256,
            learning_rate=config['learning_rate'],
            report_interval=config['report_interval'])
        self.buffer = PrioritizedReplayMemory(
            capacity=config['capacity'],
            epsilon=config['min_priority'],
            alpha=config['alpha'],
            max_priority=config['max_priority'])

        self.tensorboard_cb = CustomTensorboard(
            log_dir=config['output_directory'],
            report_interval=config['report_interval'],
            histogram_freq=1)
        self.dqn.callbacks = [self.tensorboard_cb]
        self.tensorboard_cb.register_metrics_callback(self.dqn.get_metrics)
        self.tensorboard_cb.register_metrics_callback(self.buffer.get_metrics)

        self.beta = config['beta_min']
        learner_address = config['learner_ip'] + ':' + config['starting_port']
        self._connect_sockets(learner_address)

    def _connect_sockets(self, learner_address):
        self.context = zmq.Context()
        self.parameter_socket = self.context.socket(zmq.PUB)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.bind(f'tcp://{learner_address}')
        logger.info(f'Created socket at {learner_address}')

        self.experiences_socket = self.context.socket(zmq.SUB)
        self.experiences_socket.setsockopt(zmq.LINGER, 0)
        self.experiences_socket.setsockopt(zmq.SUBSCRIBE, b'experiences')
        for ip in self.config['actors'].keys():
            for idx in range(self.config['actors'][ip]):
                port = str(int(self.config['starting_port']) + idx + 1)
                address = ip + ':' + port
                self.experiences_socket.connect(f'tcp://{address}')
                logger.info(f'Connected socket to actor at {address}')
        atexit.register(self._disconnect_sockets)

    def _disconnect_sockets(self):
        self.parameter_socket.close()
        self.experiences_socket.close()
        self.context.term()

    def update_experiences(self):
        try:
            message = self.experiences_socket.recv_multipart(flags=zmq.NOBLOCK)
            _, experiences_compressed = message[0], message[1]
            experiences_pickled = zlib.decompress(experiences_compressed)
            experiences = pickle.loads(experiences_pickled)
            for experience in experiences:
                self.buffer.add(experience.observation, experience.error)
            self.beta += (1. - self.beta
                          ) * self.config['beta_step_size'] * len(experiences)
            logger.info(
                f'Received {len(experiences)} experiences from actor. Buffer size: {self.buffer.size()}'
            )
            return True
        except zmq.Again:
            return False

    def evaluate_experiences(self):
        if not (self.buffer.size() > self.config['batch_size'] * 10):
            return
        self.tensorboard_cb.global_step += 1
        batch, indices, weights = self.buffer.sample(self.config['batch_size'],
                                                     self.beta)
        # Actual batch size can differ from self.batch_size if the memory is not filled yet
        batch_size = len(batch)

        q_values, q_values_next = self.dqn.compute_q_values(batch)
        x, y, errors = create_targets(self.input_shape, 3,
                                      self.config['gamma'], 1, batch, q_values,
                                      q_values_next)
        for idx in range(batch_size):
            self.buffer.update(indices[idx], errors[idx])
        self.dqn.train(x, y, batch_size, weights)

    def send_parameters(self):
        logger.info('Sending parameters...')
        weights = self.dqn.model.get_weights()
        p = pickle.dumps(weights, -1)
        z = zlib.compress(p)
        self.parameter_socket.send_multipart([b'parameters', z])


def main():
    with open('./config.json') as f:
        config = json.load(f)
    learner = Learner(config)
    last_parameter_update = time.time()
    while True:
        learner.update_experiences()
        learner.evaluate_experiences()
        if time.time() - last_parameter_update > 5:
            last_parameter_update = time.time()
            learner.send_parameters()


if __name__ == '__main__':
    main()