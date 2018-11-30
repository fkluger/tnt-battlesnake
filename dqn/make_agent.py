import dqn.agent as dqn
from dqn.exploration import EpsilonGreedyStrategy
from common.replay_memory import PrioritizedMemory


def make_agent(
    dqn_config,
    replay_memory_config,
    exploration_config,
    input_shape,
    num_actions: int,
    output_dir: str,
    dqn_agent_class=dqn.DQNAgent,
    double_dqn_agent_class=dqn.DoubleDQNAgent,
):
    replay_memory = PrioritizedMemory(**replay_memory_config)
    exploration_strategy = EpsilonGreedyStrategy(**exploration_config)

    hyper_parameters = dqn.HyperParameters(
        dqn_config.learning_rate,
        dqn_config.discount_factor,
        dqn_config.batch_size,
        dqn_config.importance_weight_exponent,
        dqn_config.multi_step_n,
    )

    if dqn_config.dueling:
        dqn_class = dqn.make_dqn_dueling
    else:
        dqn_class = dqn.make_dqn

    if dqn_config.double:
        agent = double_dqn_agent_class(
            target_dqn=dqn_class(
                input_shape=input_shape,
                hidden_dim=dqn_config.hidden_dim,
                num_actions=num_actions,
            ),
            target_update_rate=dqn_config.target_update_rate,
            dqn=dqn_class(
                input_shape=input_shape,
                hidden_dim=dqn_config.hidden_dim,
                num_actions=num_actions,
            ),
            replay_memory=replay_memory,
            exploration_strategy=exploration_strategy,
            hyper_parameters=hyper_parameters,
            num_actions=num_actions,
            output_dir=output_dir,
        )
    else:
        agent = dqn_agent_class(
            dqn=dqn_class(
                input_shape=input_shape,
                hidden_dim=dqn_config.hidden_dim,
                num_actions=num_actions,
            ),
            replay_memory=replay_memory,
            exploration_strategy=exploration_strategy,
            hyper_parameters=hyper_parameters,
            num_actions=num_actions,
            output_dir=output_dir,
        )
    return agent
