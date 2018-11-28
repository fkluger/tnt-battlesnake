import logging
import pickle
import numpy as np

from apex import AbstractActor
from dqn import DQN

LOGGER = logging.getLogger("DQNActor")


class Actor(AbstractActor):
    def __init__(self, config, actor_idx, starting_port, tensorboard_logger):
        super().__init__(config, actor_idx, starting_port, tensorboard_logger)
        self.dqn = DQN(
            input_shape=(config.width, config.height, config.stacked_frames),
            num_actions=3,
            learning_rate=config.learning_rate,
        )

    def _act(self, state):
        q_values = self.dqn.predict(state)
        best_action = np.argmax(q_values)
        return best_action, True

    def _update_parameters(self, message):
        online_weights_pickled, target_weights_pickled = message[1], message[2]
        online_weights = pickle.loads(online_weights_pickled)
        self.dqn.online_model.set_weights(online_weights)

        if target_weights_pickled != b"empty":
            target_weights = pickle.loads(target_weights_pickled)
            self.dqn.target_model.set_weights(target_weights)
        LOGGER.info("Received parameter update from learner.")
        return True

    def _compute_errors(self):
        _, _, errors = self.dqn.create_targets(self.buffer, len(self.buffer))
        return errors
