from typing import List
import argparse

from fabric import Connection


def get_ip(host: str):
    with Connection(host) as c:
        addresses = c.run("ifconfig |grep -Po 't addr:\K[\d.]+'").stdout.split("\n")
        return addresses[0].strip()


def start_actors(
    learner_host: str, actor_hosts: List[str], processes_per_actor: List[int], path: str
):
    print("Learner IP address is ", end=" ")
    learner_ip = get_ip(learner_host)
    for actor_index, host in enumerate(actor_hosts):
        start(
            host, path, None, learner_ip, actor_index, processes_per_actor[actor_index]
        )


def start(
    host: str,
    path: str,
    gpus: str = None,
    learner_ip: str = None,
    actor_index: int = None,
    processes: int = 1,
):
    with Connection(host) as c:
        with c.cd(path):
            if learner_ip:
                with c.prefix("conda activate bs"):
                    for process in range(processes):
                        command = f'screen -dmS "actor-{process}" "python" "run.py" "train_dqn_distributed" "--actor" "{actor_index}" "--skip_observe" "--learner_address" "{learner_ip}"'  # pylint: disable=C0301
                        print(command)
                        c.run(command)
                        print(
                            f"Starting actor {actor_index} process {process} on {host} ..."
                        )
                        c.run("sleep 1")
            else:
                with c.prefix("conda activate bs-gpu"):
                    with c.prefix(f"export CUDA_VISIBLE_DEVICES={gpus}"):
                        command = f'screen -dmS "learner" "python" "run.py" "train_dqn_distributed"'  # pylint: disable=C0301
                        print(command)
                        c.run(command)
                        print(f"Starting master on {host}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--learner",
        type=str,
        required=True,
        help="Host where the learner should be started.",
    )
    parser.add_argument(
        "-a",
        "--actors",
        required=True,
        nargs="+",
        type=str,
        help="Hosts where the actors should be started.",
    )
    parser.add_argument(
        "-p",
        "--processes",
        nargs="+",
        type=int,
        required=True,
        help="Number of processes per actor host.",
    )
    parser.add_argument("--gpus", type=str, help="CUDA_AVAILABLE_DEVICES")
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the Battlesnake root directory.",
        default="~/git/tnt-battlesnake",
    )
    args = parser.parse_args()

    if len(args.actors) != len(args.processes):
        raise ValueError(
            "A number of processes has to be specified for each actor. Length of actors and processes lists have to be equal."  # pylint: disable=C0301
        )

    start(args.learner, args.path, args.gpus)
    print("Learner started!")
    start_actors(args.learner, args.actors, args.processes, args.path)
    print("Actors started successfully!")


if __name__ == "__main__":
    main()
