import argparse
import logging

import numpy as np

from apex.configuration import Configuration
from apex.actor import Actor
from apex.enemy_actor import EnemyActor
from apex.models import Observation
from environment.battlesnake_environment import BattlesnakeEnvironment
from main_utils import wrap_main
from tensorboard_logger import TensorboardLogger


def get_args():
    parser = argparse.ArgumentParser(description="Actor for Battlesnake-DQN")
    parser.add_argument("--actor_index", type=int)
    parser.add_argument("--starting_port", type=int)
    return parser.parse_args()


def main():

    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = Configuration("./apex/config.json")

    tensorboard_logger = TensorboardLogger(config.output_directory, args.actor_index)
    actor = Actor(config, args.actor_index, args.starting_port, tensorboard_logger)
    enemy_agents = []
    for _ in range(config.snakes - 1):
        enemy_agents.append(EnemyActor(actor))
    env = BattlesnakeEnvironment(
        config,
        enemy_agents=enemy_agents,
        output_directory=f"{config.output_directory}/actor-{args.actor_index}",
        actor_idx=args.actor_index,
        tensorboard_logger=tensorboard_logger,
    )

    received_initial_parameters = False
    while not received_initial_parameters:
        received_initial_parameters = actor.update_parameters()

    while True:
        state = env.reset()
        terminal = False
        while not terminal:
            if env.stats.steps > config.random_initial_steps:
                action, greedy = actor.act(state)
            else:
                action = np.random.choice(3)
                greedy = False
            next_state, reward, terminal = env.step(action)
            actor.observe(
                Observation(
                    state, action, reward, next_state, config.discount_factor, greedy
                )
            )
            state = next_state
        if env.stats.episodes % config.parameter_update_interval == 0:
            actor.update_parameters()
        if env.stats.episodes % config.report_interval == 0:
            env.stats.report()
        if env.stats.episodes % (config.report_interval * 10) == 0:
            env.render()


if __name__ == "__main__":
    wrap_main(main)
