from typing import Union

import pickle

from common.distributed.learner import Learner

from dqn.agent.dqn_agent import DQNAgent
from dqn.agent.ddqn_agent import DoubleDQNAgent


class DQNLearner(Learner):
    def __init__(self, distributed_config, dqn_agent: Union[DQNAgent, DoubleDQNAgent]):
        super().__init__(distributed_config)
        self.received_experiences = 0
        self.dqn_agent = dqn_agent

    def _process_experiences(self, experiences):
        self.received_experiences += len(experiences)
        for experience in experiences:
            self.dqn_agent.replay_memory.add(
                experience, experience.time_difference_error
            )

    def _create_parameter_message(self):
        weights = self.dqn_agent.dqn.get_weights()
        weights_compressed = pickle.dumps(weights)
        if isinstance(self.dqn_agent, DoubleDQNAgent):
            target_weights = self.dqn_agent.target_dqn.get_weights()
            target_weights_compressed = pickle.dumps(target_weights)
            return [weights_compressed, target_weights_compressed]
        else:
            return [weights_compressed]
