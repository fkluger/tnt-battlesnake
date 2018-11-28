import keras
import numpy as np

from common.tensorflow.huber_loss import huber_loss

from .dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):

    """
    Reinforcement Learning DQN Agent that reduces the bias in the Q-value estimation by using a slowly changing \
    target network for the Q-value estimation and another (fast changing) network for the selection of the argmax.

    See https://arxiv.org/abs/1509.06461
    """

    def __init__(
        self, target_dqn: keras.Model, target_update_rate: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_dqn = target_dqn
        self.target_dqn.compile(loss=huber_loss, optimizer=self.optimizer)
        self.target_update_rate = target_update_rate
        self._update_target(soft=False)

    def _compute_loss(
        self,
        state_tensor,
        action_tensor,
        reward_tensor,
        next_state_tensor,
        non_terminal_mask,
        importance_weights,
    ):
        """
        Compute the Q-Learning loss for the Double DQN.

        Arguments:
            state_tensor {`np.ndarray`} -- Tensor of states
            action_tensor {`np.ndarray`} -- Tensor of the chosen actions
            reward_tensor {`np.ndarray`} -- Tensor of the observed rewards
            next_state_tensor {`np.ndarray`} -- Tensor of the next states
            non_terminal_mask {`np.ndarray`} -- Tensor indicating the non-terminal states
            importance_weights {`np.ndarray`} -- Tensor of importance weights

        Returns:
            `float` -- Loss
        """
        batch_size = state_tensor.shape[0]

        q_values_next = self.dqn.predict(
            [
                next_state_tensor,
                np.ones(shape=(next_state_tensor.shape[0], self.num_actions)),
            ]
        )

        q_values_next_argmax = np.zeros(shape=(batch_size,), dtype=np.int)
        q_values_next_argmax[non_terminal_mask] = np.argmax(q_values_next, axis=-1)

        q_values_next_target = np.zeros(shape=(batch_size, self.num_actions))
        q_values_next_target[non_terminal_mask] = self.target_dqn.predict(
            [
                next_state_tensor,
                np.ones(shape=(next_state_tensor.shape[0], self.num_actions)),
            ]
        )

        q_values_target = np.zeros(shape=(batch_size, self.num_actions))

        for i in range(batch_size):
            q_values_target[i, action_tensor[i]] = (
                reward_tensor[i]
                + self.hyper_parameters.gamma
                * q_values_next_target[i, q_values_next_argmax[i]]
            )

        actions_one_hot = np.eye(self.num_actions)[action_tensor]

        loss_and_time_difference_errors = self._fit(
            x=[state_tensor, actions_one_hot],
            y=q_values_target,
            importance_weights=importance_weights,
        )
        self._update_target()
        return loss_and_time_difference_errors

    def _update_target(self, soft: bool = True):
        """
        Softly update the target network using the (fast changing) other network.
        """
        if soft:
            for target_layer, layer in zip(self.target_dqn.layers, self.dqn.layers):
                target_layer.set_weights(
                    [
                        (1.0 - self.target_update_rate) * target_weights
                        + self.target_update_rate * weights
                        for target_weights, weights in zip(
                            target_layer.get_weights(), layer.get_weights()
                        )
                    ]
                )
        else:
            self.target_dqn.set_weights(self.dqn.get_weights())
