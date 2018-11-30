import pickle

from common.distributed.actor import Actor
from common.utils.compute_multi_step_rewards import compute_multi_step_rewards
from dqn.agent.dqn_agent import DQNAgent


class DQNActor(DQNAgent):
    def __init__(self, *args, **kwargs):
        self.actor: Actor = None
        super().__init__(*args, **kwargs)

    def observe(self, transitions):
        if self.hyper_parameters.multi_step_n > 1:
            transitions = compute_multi_step_rewards(
                transitions,
                self.hyper_parameters.multi_step_n,
                self.hyper_parameters.discount_factor,
            )
        self.actor.buffer.extend(transitions)
        if len(transitions) != 1 or transitions[0].next_state is None:
            self.episode += len(transitions)

        if len(self.actor.buffer) >= self.actor.max_buffer_size:
            self._compute_time_difference_errors()
            self.actor.send_experiences()

        self._update_parameters()

    def _update_parameters(self):
        parameter_message = self.actor.parameters_received()
        if parameter_message:
            weights_pickled = parameter_message[1]
            weights = pickle.loads(weights_pickled)
            self.dqn.set_weights(weights)

    def _compute_time_difference_errors(self):
        tensors = self._get_tensors(self.actor.buffer)
        _, time_difference_errors = self._compute_loss(
            *tensors, importance_weights=None
        )
        for transition, error in zip(self.actor.buffer, time_difference_errors):
            transition.time_difference_error = error
