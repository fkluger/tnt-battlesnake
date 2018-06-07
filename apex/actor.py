import atexit
import json
import pickle
import random
import zlib
import zmq
import numpy as np

from simulator.simulator import BattlesnakeSimulator
from simulator.utils import getDirection, get_next_snake_coords, is_coord_on_board, get_state_shape
from dqn import DQN
from utils import get_free_port, Experience, Observation, get_logger, get_ip_address, create_targets, ActorStatistics

logger = get_logger('Actor')


class Actor:

    buffer = list()
    max_buffer_size = 200

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
        self.epsilon = np.random.uniform(0, 0.3)
        logger.info(f'Epsilon: {self.epsilon}')
        learner_address = config['learner_ip'] + ':' + config['starting_port']
        self._connect_sockets(learner_address)

    def _connect_sockets(self, learner_address):
        self.context = zmq.Context()
        self.parameter_socket = self.context.socket(zmq.SUB)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.connect(f'tcp://{learner_address}')
        self.parameter_socket.setsockopt(zmq.SUBSCRIBE, b'parameters')
        logger.info(f'Connected socket to learner at {learner_address}')

        self.experience_socket = self.context.socket(zmq.PUB)
        self.experience_socket.setsockopt(zmq.LINGER, 0)
        port = get_free_port(int(self.config['starting_port']) + 1)
        ip_address = get_ip_address()
        self.experience_socket.bind(f'tcp://{ip_address}:{port}')
        logger.info(f'Created socket at {ip_address}:{port}')

        atexit.register(self._disconnect_sockets)

    def _disconnect_sockets(self):
        self.experience_socket.close()
        self.parameter_socket.close()
        self.context.term()

    def act(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(3)
        else:
            q_values = self.dqn.predict(state)
            best_action = np.argmax(q_values)
            logger.debug(f'Q-Values: {q_values}, Action: {best_action}')
            return best_action

    def observe(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) >= self.max_buffer_size:
            self.send_experiences()

    def update_parameters(self):
        try:
            message = self.parameter_socket.recv_multipart(flags=zmq.NOBLOCK)
            p = zlib.decompress(message[1])
            weights = pickle.loads(p)
            self.dqn.model.set_weights(weights)
            logger.info('Received parameter update from learner')
            return True
        except zmq.Again:
            return False

    def send_experiences(self):
        q_values, q_values_next = self.dqn.compute_q_values(self.buffer)
        _, _, errors = create_targets(
            input_shape=self.input_shape,
            num_actions=3,
            gamma=self.config['gamma'],
            multi_step_n=self.config['multi_step_n'],
            observations=self.buffer,
            q_values=q_values,
            q_values_next=q_values_next)
        experiences = list()
        for idx, observation in enumerate(self.buffer):
            experiences.append(Experience(observation, errors[idx]))
        experiences_pickled = pickle.dumps(experiences, -1)
        experiences_compressed = zlib.compress(experiences_pickled)
        self.experience_socket.send_multipart(
            [b'experiences', experiences_compressed])
        self.buffer.clear()


def compute_enemy_actions(env, actor):
    actions = list()
    for snake in env.state.snakes:
        if snake.id == 0:
            continue
        enemy_state = env.state.observe(snake.id)
        action = actor.act(np.expand_dims(enemy_state, -1))
        direction = getDirection(action, snake.direction)
        snake_next_body = get_next_snake_coords(snake.body, direction,
                                                env.state.fruits)
        if is_coord_on_board(snake_next_body[0], env.width, env.height):
            actions.append(action)
        else:
            actions.append(np.random.choice(3))
    return actions


def main():
    with open('./config.json') as f:
        config = json.load(f)
    actor = Actor(config)
    env = BattlesnakeSimulator(
        width=config['width'],
        height=config['height'],
        num_snakes=config['num_snakes'],
        num_fruits=config['num_fruits'],
        num_frames=config['num_frames'],
        report_interval=config['report_interval'])
    received_initial_parameters = False
    while not received_initial_parameters:
        received_initial_parameters = actor.update_parameters()
    episodes = 0
    rewards = list()
    while True:
        episode_rewards = 0
        state = env.reset()
        terminal = False
        while not terminal:
            action = actor.act(state)
            actions = [action] + compute_enemy_actions(env, actor)
            next_state, reward, terminal = env.step(actions)
            episode_rewards += reward
            if terminal:
                next_state = None
            actor.observe(Observation(state, action, reward, next_state))
            state = next_state
        episodes += 1
        rewards.append(episode_rewards)
        if episodes % 100 == 0:
            actor.update_parameters()
        if episodes % config['report_interval'] == 0:
            mean_rewards = np.mean(rewards[:-config['report_interval']])
            mean_fruits = np.mean(env.fruits_per_episode[:-config['report_interval']])
            logger.info(f'Episodes: {episodes}, Mean rewards: {mean_rewards}, Mean fruits eaten: {mean_fruits}')
            env.save_longest_episode(config['output_directory'])


if __name__ == '__main__':
    main()