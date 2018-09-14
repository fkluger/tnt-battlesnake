import time

import tensorflow as tf
import numpy as np

from apex import Configuration, Observation
from nec import NECAgent
from environment.battlesnake_environment import BattlesnakeEnvironment
from tensorboard_logger import TensorboardLogger
from main_utils import wrap_main


def main():

    with tf.Session() as sess:
        actor_index = 0
        config = Configuration("./apex/config.json")
        agent = NECAgent(config)
        tensorboard_logger = TensorboardLogger(config.output_directory, actor_index, sess.graph)
        env = BattlesnakeEnvironment(
            config,
            enemy_agents=[],
            output_directory=f"{config.output_directory}/actor-{actor_index}",
            actor_idx=actor_index,
            tensorboard_logger=tensorboard_logger,
        )

        sess.run(tf.global_variables_initializer())
        while True:
            state = env.reset()
            terminal = False
            while not terminal:
                if env.stats.steps > config.random_initial_steps:
                    if env.stats.steps % 16 == 0:
                        agent.train()
                    action, greedy = agent.act(state)
                else:
                    action = np.random.choice(3)
                    greedy = False
                next_state, reward, terminal = env.step(action)
                agent.observe(
                    Observation(
                        state,
                        action,
                        reward,
                        next_state,
                        config.discount_factor,
                        greedy,
                    )
                )
                state = next_state
            if env.stats.episodes % config.report_interval == 0:
                env.stats.report()
            if env.stats.episodes % config.render_interval == 0:
                env.render()


if __name__ == "__main__":
    wrap_main(main)
