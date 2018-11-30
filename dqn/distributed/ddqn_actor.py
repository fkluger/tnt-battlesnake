import pickle

from dqn.agent.ddqn_agent import DoubleDQNAgent
from dqn.distributed.dqn_actor import DQNActor


class DoubleDQNActor(DQNActor, DoubleDQNAgent):
    def _update_parameters(self):
        parameter_message = self.actor.parameters_received()
        if parameter_message:
            weights_pickled, target_weights_pickled = (
                parameter_message[1],
                parameter_message[2],
            )
            weights, target_weights = (
                pickle.loads(weights_pickled),
                pickle.loads(target_weights_pickled),
            )
            self.dqn.set_weights(weights)
            self.target_dqn.set_weights(target_weights)
