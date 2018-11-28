import pickle

from apex import AbstractLearner, Configuration
from dqn import DQN


class Learner(AbstractLearner):
    def __init__(self, config: Configuration):
        super().__init__(config)
        self.input_shape = config.get_input_shape()
        self.dqn = DQN(
            input_shape=self.input_shape,
            num_actions=3,
            learning_rate=config.learning_rate,
        )
        self.target_weights_changed = False

    def _create_parameters_message(self):
        online_weights = self.dqn.online_model.get_weights()
        self.stats.on_weight_export(self.dqn.online_model)
        online_weights_compressed = pickle.dumps(online_weights, -1)
        if self.target_weights_changed:
            target_weights = self.dqn.target_model.get_weights()
            target_weights_compressed = pickle.dumps(target_weights, -1)
            self.target_weights_changed = False
        else:
            target_weights_compressed = b"empty"
        return [b"parameters", online_weights_compressed, target_weights_compressed]

    def _evaluate_experiences(self):
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
