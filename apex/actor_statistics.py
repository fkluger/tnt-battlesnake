import logging

from tensorboard_logger import Metric, MetricType

LOGGER = logging.getLogger('ActorStatistics')


class ActorStatistics:

    def __init__(self, config, actor_idx, tensorboard_logger):
        self.config = config
        self.actor_idx = actor_idx
        self.tensorboard_logger = tensorboard_logger
        self.steps = 0

    def on_send(self, internal_rewards_mean, mean_inverse_losses, mean_forward_losses):
        self.tensorboard_logger.log(
            Metric(f'actor-{self.actor_idx}/mean internal rewards', MetricType.Value, internal_rewards_mean, self.steps))
        self.tensorboard_logger.log(
            Metric(f'actor-{self.actor_idx}/inverse loss', MetricType.Value, mean_inverse_losses, self.steps))
        self.tensorboard_logger.log(
            Metric(f'actor-{self.actor_idx}/forward loss', MetricType.Value, mean_forward_losses, self.steps))

    def on_observe(self, observation=None, epsilon=None):
        self.steps += 1
        if epsilon and self.steps % 10000 == 0:
            self.tensorboard_logger.log(
                Metric(f'actor-{self.actor_idx}/epsilon', MetricType.Value, epsilon, self.steps))
