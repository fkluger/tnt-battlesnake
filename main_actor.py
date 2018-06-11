import logging

import numpy as np

from apex.configuration import Configuration
from apex.actor import Actor
from apex.enemy_actor import EnemyActor
from apex.models import Observation
from environment.battlesnake_environment import BattlesnakeEnvironment


def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = Configuration('./apex/config.json')
    actor = Actor(config)
    enemy_agents = []
    for _ in range(config.snakes - 1):
        enemy_agents.append(EnemyActor(actor))
    env = BattlesnakeEnvironment(width=config.width, height=config.height,
                                 snakes=config.snakes, fruits=config.fruits, enemy_agents=enemy_agents, output_directory=config.output_directory)

    received_initial_parameters = False
    while not received_initial_parameters:
        received_initial_parameters = actor.update_parameters()

    while True:
        state = env.reset()
        terminal = False
        while not terminal:
            if env.stats.steps > config.random_initial_steps:
                action = actor.act(state)
            else:
                action = np.random.choice(3)
            next_state, reward, terminal = env.step(action)
            actor.observe(Observation(state, action, reward, next_state, config.discount_factor))
            state = next_state
        if env.stats.episodes % config.parameter_update_interval == 0:
            actor.update_parameters()
        if env.stats.episodes % config.report_interval == 0:
            env.render()


if __name__ == '__main__':
    main()
