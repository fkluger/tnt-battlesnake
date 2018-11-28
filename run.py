import argparse
import importlib

import gym_battlesnake

from sacred.observers import FileStorageObserver


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
        help="Name of the experiment's package in the ./experiments directory.",
        type=str,
    )
    args = parser.parse_args()

    experiment_module = importlib.import_module(
        f"experiments.{args.experiment_name}.main"
    )
    experiment_module.ex.observers.append(
        FileStorageObserver.create(f"./tmp/{args.experiment_name}")
    )
    experiment_module.ex.run()


if __name__ == "__main__":
    main()
