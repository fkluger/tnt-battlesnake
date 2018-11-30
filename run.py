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
    parser.add_argument(
        "--skip_observe",
        help="Whether to add an sacred observer.",
        type=bool,
        default=False,
        nargs="?",
    )
    args, _ = parser.parse_known_args()

    experiment_module = importlib.import_module(
        f"experiments.{args.experiment_name}.main"
    )
    if args.skip_observe is False:
        experiment_module.ex.observers.append(
            FileStorageObserver.create(f"./tmp/{args.experiment_name}")
        )
    experiment_module.ex.run()


if __name__ == "__main__":
    main()
