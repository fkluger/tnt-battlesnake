import os
from types import SimpleNamespace

import numpy as np
from sacred import Experiment
from tensorboardX import SummaryWriter

from common.run_episode import run_episode, run_episode_vec
from common.utils.make_environments import make_environments
from example.random_agent import RandomAgent

ex = Experiment("example")
config_path = os.path.dirname(__file__) + "/config.json"
ex.add_config(config_path)


@ex.main
def main(_run, _config):
    config = SimpleNamespace(**_config)
    environments = make_environments(config)

    output_directory = f"./tmp/example/{_run._id}/"

    writer = SummaryWriter(output_directory)

    agent = RandomAgent(num_actions=environments[0].action_space.n)

    rewards = []
    lengths = []
    for episode in range(config.episodes):
        if config.num_envs > 1:
            episode_rewards, episode_length = run_episode_vec(
                environments,
                agent,
                render=episode % config.render_episode_interval == 0,
                max_length=config.max_episode_length,
            )
        else:
            episode_rewards, episode_length = run_episode(
                environments[0],
                agent,
                render=episode % config.render_episode_interval == 0 and False,
                max_length=config.max_episode_length,
            )
        rewards.append(episode_rewards)
        lengths.append(episode_length)
        if episode % config.render_episode_interval == 0:
            mean_rewards, std_rewards = np.mean(rewards), np.std(rewards)
            mean_length, std_length = np.mean(lengths), np.std(lengths)
            global_step = config.num_envs * episode
            writer.add_scalar("rewards/mean", mean_rewards, global_step=global_step)
            writer.add_scalar(
                "rewards/standard_deviation", std_rewards, global_step=global_step
            )
            writer.add_scalar(
                "episode_length/mean", mean_length, global_step=global_step
            )
            writer.add_scalar(
                "episode_length/standard_deviation", std_length, global_step=global_step
            )
            print("Episode {}\tMean rewards {:f}".format(episode, mean_rewards))
            rewards.clear()
            lengths.clear()
