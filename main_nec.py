import time

import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np

from apex import Configuration, Observation, EnemyActor
from nec import NECAgent
from environment.battlesnake_environment import BattlesnakeEnvironment
from tensorboard_logger import TensorboardLogger, Metric, MetricType
from main_utils import wrap_main


def main():

    with tf.Session() as sess:
        actor_index = 0
        config = Configuration("./apex/config.json")
        tensorboard_logger = TensorboardLogger(
            config.output_directory, actor_index
        )
        agent = NECAgent(config, tensorboard_logger.writer)
        output_directory = f"{config.output_directory}/actor-{actor_index}"
        enemy_agents = []
        for _ in range(config.snakes - 1):
            enemy_agents.append(EnemyActor(agent))
        env = BattlesnakeEnvironment(
            config,
            enemy_agents=enemy_agents,
            output_directory=output_directory,
            actor_idx=actor_index,
            tensorboard_logger=tensorboard_logger,
        )

        projector.visualize_embeddings(tensorboard_logger.writer, agent.embedding_config)
        tensorboard_logger.writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        agent.update_indices()
        while True:
            state = env.reset()
            terminal = False
            while not terminal:
                if env.stats.steps > config.random_initial_steps:
                    if env.stats.steps % 16 == 0:
                        error, mean_q_value = agent.train()
                        tensorboard_logger.log(
                            Metric("loss", MetricType.Value, error, env.stats.steps)
                        )
                        tensorboard_logger.log(
                            Metric(
                                "mean_q_value",
                                MetricType.Value,
                                mean_q_value,
                                env.stats.steps,
                            )
                        )
                        tensorboard_logger.log(
                            Metric(
                                "epsilon",
                                MetricType.Value,
                                agent.epsilon,
                                env.stats.steps,
                            )
                        )
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
                tensorboard_logger.writer.add_summary(agent.get_summaries(), env.stats.steps)
            if env.stats.episodes % config.render_interval == 0:
                env.render()
                dnd_values = agent.get_values()
                for idx, values in enumerate(dnd_values):
                    with open(output_directory + f"/metadata_{idx}.tsv", "w") as f:
                        f.write("Index\tValue\n")
                        for index, value in enumerate(values):
                            f.write(f"{index}\t{value}\n")
                saver.save(sess, output_directory, env.stats.steps)


if __name__ == "__main__":
    wrap_main(main)
