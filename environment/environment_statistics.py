import logging

import numpy as np

from tensorboard_logger import Metric, MetricType, TensorboardLogger

LOGGER = logging.getLogger('EnvironmentStatistics')


class EnvironmentStatistics:

    last_report = 0

    steps = 0
    episodes = 0

    episode_rewards_current = 0
    episode_steps_current = 0
    episode_fruits_current = 0

    episode_rewards = list()
    episode_steps = list()
    episode_fruits = list()

    def __init__(self, output_directory, actor_idx):
        self.actor_idx = actor_idx
        self.tensorboard_logger = TensorboardLogger(output_directory)

    def report(self):
        mean_rewards = np.mean(self.episode_rewards[self.last_report:])
        mean_steps = np.mean(self.episode_steps[self.last_report:])
        mean_fruits = np.mean(self.episode_fruits[self.last_report:])
        LOGGER.info(
            f'Episodes: {self.episodes}, Steps: {self.steps}, Mean rewards: {mean_rewards}, Mean steps: {mean_steps}, Mean fruits: {mean_fruits}')
        self.tensorboard_logger.log(Metric(f'actor-{self.actor_idx}/mean rewards',
                                           MetricType.Value, mean_rewards, self.steps))
        self.tensorboard_logger.log(Metric(f'actor-{self.actor_idx}/mean episode lengths',
                                           MetricType.Value, mean_rewards, self.steps))
        self.tensorboard_logger.log(Metric(f'actor-{self.actor_idx}/mean fruits eaten',
                                           MetricType.Value, mean_rewards, self.steps))
        self.last_report = self.episodes

    def on_reset(self):
        self.episodes += 1
        self.episode_rewards.append(self.episode_rewards_current)
        self.episode_rewards_current = 0
        self.episode_steps.append(self.episode_steps_current)
        self.episode_steps_current = 0
        self.episode_fruits.append(self.episode_fruits_current)
        self.episode_fruits_current = 0

    def on_step(self, fruit_eaten, reward):
        self.steps += 1
        self.episode_steps_current += 1
        self.episode_fruits_current += fruit_eaten
        self.episode_rewards_current += reward
