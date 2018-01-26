import time
import datetime
import signal
import sys

import tensorflow as tf

from agents.dqn import DQNAgent
from agents.random import RandomAgent
from memories.prioritized_replay import PrioritizedReplayMemory
from brains.dueling_double_dqn import DuelingDoubleDQNBrain
from brains.double_dqn import DoubleDQNBrain
from runners.battlesnake_runner import SimpleRunner
from simulator.simulator import BattlesnakeSimulator
from simulator.utils import get_state_shape
from cli_args import get_args

# Suppress Traceback on Ctrl-C
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))


def main():
    args = get_args()

    shape = get_state_shape(args.width, args.height, args.frames)
    num_actions = 3

    simulator = BattlesnakeSimulator(args.width, args.height, args.snakes, args.fruits, args.frames)
    summary_writer = tf.summary.FileWriter(args.log_dir)
    brain = DuelingDoubleDQNBrain(shape, num_actions)
    memory = PrioritizedReplayMemory(200000)
    random_agent = RandomAgent(memory, num_actions)
    agent = DQNAgent(brain, memory, shape, num_actions)
    runner = SimpleRunner(random_agent, simulator)

    episodes = 0

    training = False
    print('Running {} random steps.'.format(memory.capacity))

    try:
        while training is False or episodes < args.max_episodes:

            if training is False and memory.size() == memory.capacity:
                print('Collecting random observations finished. Beginning training...')
                runner.agent = agent
                training = True

            episodes += 1
            runner.run()

            if training is True and episodes % args.report_interval == 0:
                mean_episode_length = sum(runner.episode_lengths[-args.report_interval:]) * 1.0 / args.report_interval
                mean_episode_rewards = sum(runner.episode_rewards[-args.report_interval:]) * 1.0 / args.report_interval
                mean_loss = sum(runner.losses[-args.report_interval:]) * 1.0 / args.report_interval
                mean_q_value_estimates = sum(
                    runner.q_value_estimates[-args.report_interval:]) * 1.0 / args.report_interval
                ts = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y %H:%M:%S')
                print('{} - Episode: {}\tSteps: {}\tMean reward: {:4.4f}\tMean length: {:4.4f}'.format(ts,
                                                                                                       episodes, runner.steps, mean_episode_rewards, mean_episode_length))
                metrics = [
                    {'name': 'mean rewards', 'value': mean_episode_rewards},
                    {'name': 'mean episode length', 'value': mean_episode_length},
                    {'name': 'mean loss', 'value': mean_loss},
                    {'name': 'mean q value estimates', 'value': mean_q_value_estimates}
                ]

                metrics.extend(agent.get_metrics())

                write_summary(summary_writer, runner.steps, metrics)

            if training is True and episodes % (args.report_interval * 50) == 0:
                simulator.save_longest_episode()
        summary_writer.close()
    finally:
        brain.model.save('{}-model.h5'.format(episodes))
        # TODO: Persist episode count, steps, epsilon, ...


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
