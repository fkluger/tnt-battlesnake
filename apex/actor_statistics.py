import logging
from collections import deque

from tensorboard_logger import Metric, MetricType

LOGGER = logging.getLogger('ActorStatistics')


class ActorStatistics:

    def __init__(self, config, actor_idx, tensorboard_logger):
        self.config = config
        self.actor_idx = actor_idx
        self.tensorboard_logger = tensorboard_logger
        self.actions_taken = deque([], 10000)
        self.steps = 0

    def on_observe(self, observation):
        self.actions_taken.append(observation.action)
        self.steps += 1
