import time
import datetime
import signal
import sys
import os
import json

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


def get_time_string():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y %H-%M-%S')


def main():
    args = get_args()

    args = vars(args)

    shape = get_state_shape(args['width'], args['height'], args['frames'])
    num_actions = 3

    if args['continue_experiment'] is not None:
        print('Continuing experiment {}'.format(args['continue_experiment']))
        output_directory = args['continue_experiment']
        with open('{}/parameters.json'.format(output_directory)) as f:
            parameters = json.load(f)
            args = {**args, **parameters}
            args['continue_experiment'] = output_directory
    else:
        output_directory = 'experiment - ' + get_time_string()
        os.makedirs(output_directory)

        with open('{}/parameters.json'.format(output_directory), 'w') as f:
            json.dump(args, f, indent=2)

    simulator = BattlesnakeSimulator(args['width'], args['height'], args['snakes'], args['fruits'], args['frames'])
    brain = DuelingDoubleDQNBrain(shape, num_actions, args['learning_rate'])
    memory = PrioritizedReplayMemory(args['replay_capacity'], args['replay_min_prio'],
                                     args['replay_alpha_prio'], args['replay_max_prio'])
    random_agent = RandomAgent(memory, num_actions)
    agent = DQNAgent(brain, memory, shape, num_actions, args['gamma'], args['epsilon_max'],
                     args['epsilon_min'], args['epsilon_lambda'], args['batch_size'], args['target_update_freq'], args['replay_beta_min'])
    runner = SimpleRunner(random_agent, simulator)

    summary_writer = tf.summary.FileWriter(output_directory)

    episodes = 0

    if args['continue_experiment'] is not None:
        checkpoint_files = os.listdir(output_directory)
        checkpoint_files = [checkpoint_file for checkpoint_file in checkpoint_files if checkpoint_file.endswith('-checkpoint.json')]
        if checkpoint_files:
            checkpoint_files = sorted(checkpoint_files, reverse=True)
            checkpoint_file = checkpoint_files[0]
            with open(f'{output_directory}/{checkpoint_file}') as f:
                checkpoint = json.load(f)
                print('Restoring checkpoint file {} with content {}.'.format(checkpoint_file, checkpoint))
                episodes = checkpoint['episodes']
                runner.steps, agent.steps = checkpoint['steps'], checkpoint['steps']
                agent.epsilon = checkpoint['epsilon']
                agent.beta = checkpoint['beta']
                weight_file = checkpoint_file.replace('checkpoint.json', 'model.h5')
                brain.model.load_weights(f'{output_directory}/{weight_file}')
                brain.target_model.load_weights(f'{output_directory}/{weight_file}')
                memory.max_priority = checkpoint['replay_max_prio']

    training = False
    print('Running {} random steps.'.format(memory.capacity))

    try:
        while training is False or episodes < args['max_episodes']:

            if training is False and memory.size() == memory.capacity:
                print('Collecting random observations finished. Beginning training...')
                runner.agent = agent
                training = True

            episodes += 1
            runner.run()

            if training is True and episodes % args['report_interval'] == 0:
                mean_episode_length = sum(
                    runner.episode_lengths[-args['report_interval']:]) * 1.0 / args['report_interval']
                mean_episode_rewards = sum(
                    runner.episode_rewards[-args['report_interval']:]) * 1.0 / args['report_interval']
                mean_loss = sum(runner.losses[-args['report_interval']:]) * 1.0 / args['report_interval']
                mean_q_value_estimates = sum(
                    runner.q_value_estimates[-args['report_interval']:]) * 1.0 / args['report_interval']
                print('{} - Episode: {}\tSteps: {}\tMean reward: {:4.4f}\tMean length: {:4.4f}'.format(get_time_string(),
                                                                                                       episodes, runner.steps, mean_episode_rewards, mean_episode_length))
                metrics = [
                    {'name': 'mean rewards', 'value': mean_episode_rewards},
                    {'name': 'mean episode length', 'value': mean_episode_length},
                    {'name': 'mean loss', 'value': mean_loss},
                    {'name': 'mean q value estimates', 'value': mean_q_value_estimates}
                ]

                metrics.extend(agent.get_metrics())

                write_summary(summary_writer, runner.steps, metrics)

            if training is True and episodes % (args['report_interval'] * 50) == 0:
                simulator.save_longest_episode(output_directory)
            if training is True and episodes % (args['report_interval'] * 100) == 0:
                brain.model.save_weights('{}/{}-model.h5'.format(output_directory, episodes))
                with open('{}/{}-checkpoint.json'.format(output_directory, episodes), 'w') as f:
                    checkpoint = {
                        'episodes': episodes,
                        'steps': runner.steps,
                        'epsilon': agent.epsilon,
                        'beta': agent.beta,
                        'replay_max_prio': memory.max_priority
                    }
                    json.dump(checkpoint, f, indent=2)
        summary_writer.close()
    finally:
        brain.model.save_weights('{}/{}-model.h5'.format(output_directory, episodes))


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
