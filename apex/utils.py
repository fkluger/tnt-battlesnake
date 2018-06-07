import logging
import socket
from contextlib import closing

import numpy as np


def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)


logger = get_logger('Battlesnake')


class ActorStatistics:
    def __init__(self, mean_rewards):
        self.mean_rewards = mean_rewards


class Experience:
    def __init__(self, observation, error):
        self.observation = observation
        self.error = error


class Observation:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state


def get_free_port(starting_port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        for i in range(200):
            port = starting_port + i
            try:
                s.bind(('', port))
                return s.getsockname()[1]
            except Exception:
                logger.debug(
                    f'Port {port} is already in use. Trying next port...')


def get_ip_address():
    return socket.gethostbyname(socket.gethostname())


def create_targets(input_shape, num_actions, gamma, multi_step_n, observations,
                   q_values, q_values_next):
    batch_size = len(observations)
    x = np.zeros((batch_size, ) + input_shape)
    y = np.zeros((batch_size, num_actions))
    errors = np.zeros(batch_size)
    for idx, o in enumerate(observations):
        target = q_values[idx]
        target_old = np.copy(target)
        if o.next_state is None:
            target[o.action] = o.reward
        else:
            best_action = np.argmax(q_values[idx])
            target[o.action] = o.reward + (
                gamma**multi_step_n) * q_values_next[idx, best_action]
        if o.state is None:
            print(o)
        x[idx] = o.state
        y[idx] = target
        errors[idx] = np.abs(target[o.action] - target_old[o.action])
    return x, y, errors