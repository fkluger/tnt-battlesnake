import argparse
import time
import datetime


def get_args():
    parser = argparse.ArgumentParser(description='Train a DQN for Battlesnake')

    parser.add_argument('--max_episodes', default=200e6, help='Number of episodes to run.')
    parser.add_argument('--report_interval', default=10, help='Interval of the reports.')

    parser.add_argument('--width', default=20, help='Width of the board.')
    parser.add_argument('--height', default=20, help='Height of the board.')
    parser.add_argument('--snakes', default=1, help='Number of snakes on the board.')
    parser.add_argument('--fruits', default=3, help='Number of fruits on the board.')

    parser.add_argument('--frames', default=2, help='Frames of the game to concatenate.')

    parser.add_argument('--gamma', default=0.9, help='Discount factor for Bellman update.')
    parser.add_argument('--epsilon-max', default=1.0, help='Start value for epsilon greedy exploration.')
    parser.add_argument('--epsilon-min', default=0.1, help='End value for epsilon greedy exploration.')
    parser.add_argument('--lambda', default=1e-4,
                        help='Decay rate for epsilon greedy exploration. Example: 1e-4 means that epsilon is cut in half every 1e4 steps.')
    parser.add_argument('--batch-size', default=32, help='Batch size that the DQN is trained on.')
    parser.add_argument('--learning-rate', default=0.00025, help='Learning rate of the RMSprop optimizer.')

    parser.add_argument('--target-update-freq', default=10000,
                        help='Double DQN target network parameter update frequency.')

    parser.add_argument('--replay-capacity', default=200000, help='Capacity of the replay memory.')

    parser.add_argument('--replay-alpha-prio', default=0.9,
                        help='Degree of prioritization for prioritized experience replay. 0 means all experiences have the same priority.')
    parser.add_argument('--replay-min-prio', default=0.01, help='Minimum priority for prioritized experience replay.')
    parser.add_argument('--replay-max-prio', default=1,
                        help='Initial maximum priority for prioritized experience replay.')

    return parser.parse_args()
