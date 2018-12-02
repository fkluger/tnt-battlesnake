import os
import time
from types import SimpleNamespace

import gym
import numpy as np
from tensorboardX import SummaryWriter

from common.run_episode import run_episode, run_episode_vec
from common.distributed.actor import Actor
from dqn.distributed import DQNActor, DoubleDQNActor
from dqn.make_agent import make_agent
from gym_battlesnake.wrappers import FrameStack


def is_int(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_run_id(base_dir: str):
    run_ids = []
    while not run_ids:
        dirnames = [
            dirname
            for dirname in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, dirname))
        ]
        run_ids = sorted([int(dirname) for dirname in dirnames if is_int(dirname)])
    return run_ids[-1]


def main_actor(run_id, config, actor: int):
    environments = []
    for _ in range(config.num_envs):
        environment = gym.make(config.env)
        if config.frame_stack > 1:
            environment = FrameStack(environment, num_stacked_frames=config.frame_stack)
        environments.append(environment)

    input_shape = environments[0].observation_space.shape
    num_actions = environments[0].action_space.n

    base_directory = "./tmp/train_dqn_distributed"

    run_id = get_run_id(base_directory)

    output_directory = f"{base_directory}/{run_id}/actor_{actor}"

    writer = SummaryWriter(output_directory)

    config.exploration["epsilon_max"] = config.exploration["epsilon_max"] ** (
        (actor / 8) * 7
    )
    config.exploration["epsilon_min"] = config.exploration["epsilon_max"]

    agent = make_agent(
        SimpleNamespace(**config.dqn),
        config.replay_memory,
        config.exploration,
        input_shape,
        num_actions,
        output_directory,
        DQNActor,
        DoubleDQNActor,
    )

    agent.actor = Actor(SimpleNamespace(**config.distributed))

    rewards = []
    lengths = []
    last_log_ts = 0
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
                render=episode % config.render_episode_interval == 0,
                max_length=config.max_episode_length,
            )
        rewards.append(episode_rewards)
        lengths.append(episode_length)

        if time.time() - last_log_ts > config.logging_interval:
            last_log_ts = time.time()
            mean_rewards, std_rewards = np.mean(rewards), np.std(rewards)
            mean_length, std_length = np.mean(lengths), np.std(lengths)
            global_step = config.num_envs * episode
            writer.add_scalar(
                "actor/rewards/mean", mean_rewards, global_step=global_step
            )
            writer.add_scalar(
                "actor/rewards/standard_deviation", std_rewards, global_step=global_step
            )
            writer.add_scalar(
                "actor/episode_length/mean", mean_length, global_step=global_step
            )
            writer.add_scalar(
                "actor/episode_length/standard_deviation",
                std_length,
                global_step=global_step,
            )
            writer.add_scalar(
                "actor/dqn/epsilon",
                agent.exploration_strategy.epsilon,
                global_step=global_step,
            )
            print(
                "Episode {}\tMean rewards {:f}\tEpsilon {:f}".format(
                    episode, mean_rewards, agent.exploration_strategy.epsilon
                )
            )
            rewards.clear()
            lengths.clear()
