import time
import datetime
import signal
import sys

import tensorflow as tf

from agents.dqn import DQNAgent
from memories.replay import ReplayMemory
from brains.plain_dqn import PlainDQNBrain
from runners.battlesnake_runner import SimpleRunner
from simulator.simulator import BattlesnakeSimulator
from simulator.utils import get_state_shape
from cli_args import get_args

# Suppress Traceback on Ctrl-C
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))


def main():
    args = get_args()

    shape = get_state_shape(args.width, args.height, args.frames)

    simulator = BattlesnakeSimulator(args.width, args.height, args.snakes, args.fruits, args.frames)
    summary_writer = tf.summary.FileWriter(args.log_dir)
    brain = PlainDQNBrain(shape, 3)
    memory = ReplayMemory()
    agent = DQNAgent(brain, memory, summary_writer, shape, 3)
    runner = SimpleRunner(agent, simulator)

    episodes = 0

    while episodes < args.max_episodes:
        episodes += 1
        runner.run()

        if episodes % args.report_interval == 0:
            mean_episode_length = sum(runner.episode_lengths[-args.report_interval:]) * 1.0 / args.report_interval
            mean_episode_rewards = sum(runner.episode_rewards[-args.report_interval:]) * 1.0 / args.report_interval
            mean_loss = sum(runner.losses[-args.report_interval:]) * 1.0 / args.report_interval
            mean_q_value_estimates = sum(runner.q_value_estimates[-args.report_interval:]) * 1.0 / args.report_interval
            ts = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y %H:%M:%S')
            print('{} - Episode: {}\tMean reward: {:4.4f}\tMean length: {:4.4f}'.format(ts,
                                                                                        episodes, mean_episode_rewards, mean_episode_length))
            metrics = [
                {'name': 'mean rewards', 'value': mean_episode_rewards},
                {'name': 'mean episode length', 'value': mean_episode_length},
                {'name': 'mean loss', 'value': mean_loss},
                {'name': 'mean q value estimates', 'value': mean_q_value_estimates}
            ]

            write_summary(summary_writer, agent.steps, metrics)

        if episodes % (args.report_interval * 100) == 0:
            simulator.save_longest_episode()
    summary_writer.close()


def write_summary(summary_writer, steps, metrics):
    for metric in metrics:
        summary_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(
                tag=metric['name'],
                simple_value=metric['value'])
        ]), global_step=steps)
    summary_writer.flush()


if __name__ == '__main__':
    main()
