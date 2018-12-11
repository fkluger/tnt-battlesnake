import argparse
import importlib

import gym_battlesnake

from sacred.observers import FileStorageObserver
from sacred.arg_parser import get_config_updates


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
    args, unknown_args = parser.parse_known_args()

    config_updates, _ = get_config_updates(
        [arg for arg in unknown_args if arg != "with"]
    )

    experiment_module = importlib.import_module(
        f"experiments.{args.experiment_name}.main"
    )
    if args.skip_observe is False:
        experiment_module.ex.observers.append(
            FileStorageObserver.create(f"./tmp/{args.experiment_name}")
        )
    experiment_module.ex.run(config_updates=config_updates)


if __name__ == "__main__":
    main()
