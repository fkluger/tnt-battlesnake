import argparse
import time
import datetime


def get_args():
    parser = argparse.ArgumentParser(description='Train a DQN for Battlesnake')

    parser.add_argument('--max_episodes', default=200e6, help='Number of episodes to run')
    parser.add_argument('--report_interval', default=10, help='Interval of the reports')
    parser.add_argument('--log_dir', default='./stats', help='Log directory')

    parser.add_argument('--width', default=10, help='Width of the board')
    parser.add_argument('--height', default=10, help='Height of the board')
    parser.add_argument('--snakes', default=1, help='Number of snakes on the board')
    parser.add_argument('--fruits', default=1, help='Number of fruits on the board')

    parser.add_argument('--frames', default=1, help='Frames of the game to concatenate')

    return parser.parse_args()
