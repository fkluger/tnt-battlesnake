import logging

LOGGER = logging.getLogger('ActorStatistics')


class ActorStatistics:

    def __init__(self, config, actor_idx, tensorboard_logger):
        self.config = config
        self.actor_idx = actor_idx
        self.tensorboard_logger = tensorboard_logger
        self.steps = 0

    def on_observe(self, observation):
        self.steps += 1
