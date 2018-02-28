import time
import datetime
import signal
import sys
import os
import json

import tensorflow as tf
import numpy as np

from agents.dqn import DQNAgent
from agents.random import RandomAgent
from agents.distributional_dqn import DistributionalDQNAgent
from memories.prioritized_replay import PrioritizedReplayMemory
from brains.dueling_double_dqn import DuelingDoubleDQNBrain
from brains.distributional_dueling_double_dqn import DistributionalDuelingDoubleDQNBrain
from brains.double_dqn import DoubleDQNBrain
from runners.battlesnake_runner import SimpleRunner
from simulator.simulator import BattlesnakeSimulator
from simulator.utils import get_state_shape
from cli_args import get_args
from tensorboard import CustomTensorboard

# Suppress Traceback on Ctrl-C
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))


def get_time_string():
    return datetime.datetime.fromtimestamp(
        time.time()).strftime('%d.%m.%Y %H-%M-%S')


def main():
    args = get_args()

    args = vars(args)

    shape = get_state_shape(args['width'], args['height'], args['frames'])
    num_actions = 3

    if args['continue_experiment'] is not None:
        print('Continuing experiment {}'.format(args['continue_experiment']))
        output_directory = args['continue_experiment']
        with open(f'{output_directory}/parameters.json') as f:
            parameters = json.load(f)
            args = {**args, **parameters}
            args['continue_experiment'] = output_directory
    else:
        output_directory = 'experiment - ' + get_time_string()
        os.makedirs(output_directory)

        with open('{}/parameters.json'.format(output_directory), 'w') as f:
            json.dump(args, f, indent=2)

    simulator = BattlesnakeSimulator(args['width'], args['height'],
                                     args['snakes'], args['fruits'],
                                     args['frames'], args['report_interval'])

    if args["distributional"]:
        brain = DistributionalDuelingDoubleDQNBrain(
            num_quantiles=args['num_quantiles'],
            input_shape=shape,
            num_actions=num_actions,
            learning_rate=args['learning_rate'],
            report_interval=args['report_interval'])
    else:
        brain = DuelingDoubleDQNBrain(
            input_shape=shape,
            num_actions=num_actions,
            learning_rate=args['learning_rate'])

    memory = PrioritizedReplayMemory(
        args['replay_capacity'], args['replay_min_prio'],
        args['replay_alpha_prio'], args['replay_max_prio'])
    random_agent = RandomAgent(memory, num_actions)
    if args["distributional"]:
        agent = DistributionalDQNAgent(
            num_quantiles=args['num_quantiles'],
            brain=brain,
            memory=memory,
            input_shape=shape,
            num_actions=num_actions,
            GAMMA=args['gamma'],
            EPSILON_MAX=args['epsilon_max'],
            EPSILON_MIN=args['epsilon_min'],
            LAMBDA=args['epsilon_lambda'],
            batch_size=args['batch_size'],
            update_target_freq=args['target_update_freq'],
            replay_beta_min=args['replay_beta_min'],
            multi_step_n=args['multi_step_n'])
    else:
        agent = DQNAgent(
            brain=brain,
            memory=memory,
            input_shape=shape,
            num_actions=num_actions,
            GAMMA=args['gamma'],
            EPSILON_MAX=args['epsilon_max'],
            EPSILON_MIN=args['epsilon_min'],
            LAMBDA=args['epsilon_lambda'],
            batch_size=args['batch_size'],
            update_target_freq=args['target_update_freq'],
            replay_beta_min=args['replay_beta_min'],
            multi_step_n=args['multi_step_n'])

    tensorboard_cb = CustomTensorboard(log_dir=output_directory, report_interval=args['report_interval'], histogram_freq=1)
    runner = SimpleRunner(random_agent, simulator, args['training_interval'],
                          args['report_interval'], tensorboard_cb)


    tensorboard_cb.register_metrics_callback(agent.get_metrics)
    tensorboard_cb.register_metrics_callback(memory.get_metrics)
    tensorboard_cb.register_metrics_callback(simulator.get_metrics)
    tensorboard_cb.register_metrics_callback(runner.get_metrics)

    brain.set_callbacks([tensorboard_cb])

    episodes = 0

    def sort_checkpoint_files(f):
        return int(f.split('-')[0])

    if args['continue_experiment'] is not None:
        checkpoint_files = os.listdir(output_directory)
        checkpoint_files = [
            checkpoint_file for checkpoint_file in checkpoint_files
            if checkpoint_file.endswith('-checkpoint.json')
        ]
        if checkpoint_files:
            checkpoint_files = sorted(
                checkpoint_files, key=sort_checkpoint_files, reverse=True)
            checkpoint_file = checkpoint_files[0]
            with open(f'{output_directory}/{checkpoint_file}') as f:
                checkpoint = json.load(f)
                print('Restoring checkpoint file {} with content {}.'.format(
                    checkpoint_file, checkpoint))
                episodes, simulator.episodes = checkpoint[
                    'episodes'], checkpoint['episodes']
                runner.steps, agent.steps, simulator.steps = checkpoint[
                    'steps'], checkpoint['steps'], checkpoint['steps']
                agent.epsilon = checkpoint['epsilon']
                agent.beta = checkpoint['beta']
                weight_file = checkpoint_file.replace('checkpoint.json',
                                                      'model.h5')
                brain.model.load_weights(f'{output_directory}/{weight_file}')
                brain.target_model.load_weights(
                    f'{output_directory}/{weight_file}')
                memory.max_priority = checkpoint['replay_max_prio']

    training = False
    print('Running {} random steps.'.format(80000))

    while training is False or episodes < args['max_episodes']:

        if training is False and memory.size() > 80000:
            print(
                'Collecting random observations finished. Beginning training...'
            )
            runner.agent = agent
            training = True

        episodes += 1
        runner.run()

        if training is False and episodes % 1000 == 0:
            print(f'Random runs {memory.size() * 100 / 80000}% complete.')

        if training is True and episodes % (args['report_interval'] * 50) == 0:
            simulator.save_longest_episode(output_directory)
        if training is True and episodes % (
                args['report_interval'] * 100) == 0:
            brain.model.save_weights('{}/{}-model.h5'.format(
                output_directory, episodes))
            with open('{}/{}-checkpoint.json'.format(output_directory,
                                                     episodes), 'w') as f:
                checkpoint = {
                    'episodes': episodes,
                    'steps': runner.steps,
                    'epsilon': agent.epsilon,
                    'beta': agent.beta,
                    'replay_max_prio': memory.max_priority
                }
                json.dump(checkpoint, f, indent=2)


if __name__ == '__main__':
    main()
