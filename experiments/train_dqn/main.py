import os
from types import SimpleNamespace

import gym
import numpy as np
from sacred import Experiment
from tensorboardX import SummaryWriter

from common.run_episode import run_episode
from dqn.make_agent import make_agent

ex = Experiment("train_dqn")
config_path = os.path.dirname(__file__) + "/config.json"
ex.add_config(config_path)


@ex.main
def main(_run, _config):
    config = SimpleNamespace(**_config)
    environment = gym.make(config.env)

    input_shape = environment.observation_space.shape
    num_actions = environment.action_space.n

    output_directory = f"./tmp/train_dqn/{_run._id}/"

    writer = SummaryWriter(output_directory)

    agent = make_agent(
        SimpleNamespace(**config.dqn),
        config.replay_memory,
        config.exploration,
        input_shape,
        num_actions,
        output_directory,
    )

    rewards = []
    for episode in range(config.episodes):
        episode_rewards = run_episode(
            environment,
            agent,
            render=episode % config.render_episode_interval == 0,
            max_length=config.max_episode_length,
        )
        rewards.append(episode_rewards)
        if episode % config.training_interval == 0:
            for _ in range(config.training_interval):
                loss = agent.train()

            if loss and episode % (config.training_interval * 10) == 0:
                mean_rewards = np.mean(rewards)
                std_rewards = np.std(rewards)
                writer.add_scalar("dqn/loss", loss, global_step=episode)
                writer.add_scalar("rewards/mean", mean_rewards, global_step=episode)
                writer.add_scalar(
                    "rewards/standard_deviation", std_rewards, global_step=episode
                )
                writer.add_scalar(
                    "dqn/epsilon",
                    agent.exploration_strategy.epsilon,
                    global_step=episode,
                )
                print(
                    "Episode {}\tMean rewards {:f}\tLoss {:f}\tEpsilon {:f}".format(
                        episode, mean_rewards, loss, agent.exploration_strategy.epsilon
                    )
                )
                rewards.clear()
