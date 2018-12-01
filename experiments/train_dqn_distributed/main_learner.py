from types import SimpleNamespace
import time

import gym
import numpy as np
from tensorboardX import SummaryWriter

from dqn.make_agent import make_agent
from dqn.distributed import DQNLearner
from gym_battlesnake.wrappers import FrameStack


def main_learner(run_id, config):
    environments = []
    for _ in range(config.num_envs):
        environment = gym.make(config.env)
        if config.frame_stack > 1:
            environment = FrameStack(environment, num_stacked_frames=config.frame_stack)
        environments.append(environment)

    input_shape = environments[0].observation_space.shape
    num_actions = environments[0].action_space.n

    output_directory = f"./tmp/train_dqn_distributed/{run_id}/"

    writer = SummaryWriter(output_directory)

    learner = DQNLearner(
        distributed_config=SimpleNamespace(**config.distributed),
        dqn_agent=make_agent(
            SimpleNamespace(**config.dqn),
            config.replay_memory,
            config.exploration,
            input_shape,
            num_actions,
            output_directory,
        ),
    )

    losses = []
    training_steps = 0
    last_log_ts = 0
    last_received_experiences = 0
    last_training_steps = 0
    while True:
        learner.receive_experiences()
        loss = learner.dqn_agent.train()
        if loss is not None:
            training_steps += learner.dqn_agent.hyper_parameters.batches
            losses.append(loss)
            if training_steps % 100 == 0:
                learner.send_parameters()

        if time.time() - last_log_ts > config.logging_interval and losses:
            last_log_ts = time.time()
            mean_loss = np.mean(losses)
            writer.add_scalar(
                "learner/loss/mean", mean_loss, global_step=training_steps
            )
            writer.add_scalar(
                "learner/loss/std", np.std(losses), global_step=training_steps
            )

            writer.add_histogram(
                "learner/memory/mean_samples",
                learner.dqn_agent.replay_memory.tree.sampling_counter,
                global_step=training_steps,
            )

            writer.add_histogram(
                "learner/memory/mean_priorities",
                learner.dqn_agent.replay_memory.tree.priorities,
                global_step=training_steps,
            )

            writer.add_scalar(
                "learner/experiences_per_second",
                (learner.received_experiences - last_received_experiences)
                / config.logging_interval,
                global_step=training_steps,
            )
            writer.add_scalar(
                "learner/train_steps_per_second",
                (training_steps - last_training_steps) / config.logging_interval,
                global_step=training_steps,
            )
            last_training_steps = training_steps
            last_received_experiences = learner.received_experiences
            print(f"Training step: {training_steps}\tLoss: {mean_loss}")
            losses.clear()
