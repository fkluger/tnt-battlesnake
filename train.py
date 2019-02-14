import argparse
import os

import tensorflow as tf
import ray
import ray.tune as tune
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph

from gym_battlesnake.envs import BattlesnakeEnv


def env_creator(config):
    env = BattlesnakeEnv(
        width=config["width"],
        height=config["height"],
        num_snakes=config["num_snakes"],
        stacked_frames=config["stacked_frames"],
    )
    return env


class BattlesnakeVisionNet(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict["obs"]

        with tf.name_scope("battlesnake_vision_net"):
            hidden = tf.layers.Conv2D(32, 1, 1, activation=tf.nn.leaky_relu)(inputs)
            hidden = tf.layers.Conv2D(64, 2, 2, activation=tf.nn.leaky_relu)(hidden)
            hidden = tf.layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu)(hidden)
            hidden = tf.layers.Flatten()(hidden)
            last_layer = tf.layers.Dense(512, activation=tf.nn.leaky_relu)(hidden)
            output = tf.layers.Dense(num_outputs)(last_layer)
            return output, last_layer


def register():
    ray.tune.register_env("battlesnake", env_creator)
    ModelCatalog.register_custom_model("battlesnake_vision_net", BattlesnakeVisionNet)


def on_train_result(info):
    iterations = info["result"]["iterations_since_restore"]
    if iterations % 25 == 0:
        agent = info["agent"]
        agent.export_policy_model(
            os.path.join(agent.logdir, f"model_{iterations}"), "snake_0"
        )


def get_agent_config(
    width: int = 9,
    height: int = 9,
    stacked_frames: int = 2,
    num_snakes: int = 3,
    num_workers: int = 3,
):
    env_config = {
        "width": width,
        "height": height,
        "num_snakes": num_snakes,
        "stacked_frames": stacked_frames,
    }
    env = env_creator(env_config)
    agent_config = {
        "model": {"custom_model": "battlesnake_vision_net"},
        "env_config": env_config,
        "num_workers": num_workers,
        "num_envs_per_worker": 32,
        "double_q": False,
        "num_atoms": 51,
        "v_min": -2.0,
        "v_max": env_config["width"] ** 2.0,
        "buffer_size": 1_000_000,
        "noisy": True,
        "prioritized_replay_alpha": 0.5,
        "beta_annealing_fraction": 1.0,
        "final_prioritized_replay_beta": 1.0,
        "callbacks": {"on_train_result": tune.function(on_train_result)},
        "multiagent": {
            "policy_graphs": {
                "snake_0": (DQNPolicyGraph, env.obs_space, env.action_space, {})
            },
            "policy_mapping_fn": tune.function(lambda agent_id: "snake_0"),
        },
    }
    return agent_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", help="Name of the RL algorithm.", type=str, default="DQN"
    )
    parser.add_argument(
        "--workers", help="Number of worker processes.", type=int, default=3
    )
    parser.add_argument(
        "--frames", help="Number of stacked frames.", type=int, default=2
    )
    parser.add_argument(
        "--size", help="Width/Height of the game.", type=int, default=13
    )
    parser.add_argument("--snakes", help="Number of snakes.", type=int, default=3)
    args, _ = parser.parse_known_args()
    ray.init()
    register()
    tune.run_experiments(
        {
            "battlesnake": {
                "run": args.algorithm,
                "env": "battlesnake",
                "config": get_agent_config(
                    width=args.size,
                    height=args.size,
                    stacked_frames=args.frames,
                    num_snakes=args.snakes,
                    num_workers=args.workers,
                ),
            }
        }
    )


if __name__ == "__main__":
    main()
