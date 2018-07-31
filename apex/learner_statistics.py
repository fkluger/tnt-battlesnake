import json
import logging
import time

import numpy as np

from tensorboard_logger import Metric, MetricType

LOGGER = logging.getLogger('LearnerStatistics')


class LearnerStatistics:

    def __init__(self, config, tensorboard_logger, buffer):
        self.config = config
        self.tensorboard_logger = tensorboard_logger
        with open(f'{self.tensorboard_logger.output_directory}/config.json', 'w') as f:
            json.dump(self.config._config, f)
        self.buffer = buffer
        self.received_batches = 0
        self.received_observations = 0
        self.last_batch_timestamp = time.time()
        self.training_counter = 0

    def on_batch_receive(self, experiences):
        self.received_batches += 1
        self.received_observations += len(experiences)

    def on_evaluation(self, batch, errors, loss, icm_loss):
        self.training_counter += len(batch)
        time_difference = time.time() - self.last_batch_timestamp

        global_step = self.received_observations / self.config.get_num_actors()
        if time_difference > 15:
            self.last_batch_timestamp = time.time()
            if self.config.icm:
                self.tensorboard_logger.log(Metric('learner/icm loss', MetricType.Value, icm_loss, global_step))
            self.tensorboard_logger.log(Metric('learner/batch loss', MetricType.Value, loss, global_step))
            self.tensorboard_logger.log(Metric('learner/batch mean error',
                                               MetricType.Value, np.mean(errors), global_step))
            self.tensorboard_logger.log(Metric('learner/samples per second', MetricType.Value,
                                               self.training_counter / time_difference, global_step))
            self.training_counter = 0
            self.tensorboard_logger.log(Metric('learner/priorities', MetricType.Histogram,
                                               self.buffer.tree.priorities, global_step))
