import argparse
import os

import tensorflow as tf
import ray
import ray.tune as tune
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.agents.impala.vtrace_policy_graph import VTracePolicyGraph

from gym_battlesnake.envs import BattlesnakeEnv


def env_creator(config):
    env = BattlesnakeEnv(
        width=config["width"],
        height=config["height"],
        num_snakes=config["num_snakes"],
        stacked_frames=config["stacked_frames"],
    )
    return env


class BattlesnakeResNet(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict["obs"]

        with tf.variable_scope("battlesnake_resnet"):
            conv_out = inputs / 255.0
            for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
                conv_out = tf.layers.Conv2D(num_ch, 1, strides=1, padding="same")(
                    conv_out
                )
                conv_out = tf.layers.BatchNormalization()(conv_out)
                conv_out = tf.nn.pool(
                    conv_out,
                    window_shape=[3, 3],
                    pooling_type="MAX",
                    padding="SAME",
                    strides=[2, 2],
                )
                # Residual block(s).
                for j in range(num_blocks):
                    with tf.variable_scope("residual_%d_%d" % (i, j)):
                        block_input = conv_out
                        conv_out = tf.nn.relu(conv_out)
                        conv_out = tf.layers.Conv2D(
                            num_ch, 3, strides=1, padding="same"
                        )(conv_out)
                        conv_out = tf.layers.BatchNormalization()(conv_out)
                        conv_out = tf.nn.relu(conv_out)
                        conv_out = tf.layers.Conv2D(
                            num_ch, 3, strides=1, padding="same"
                        )(conv_out)
                        conv_out = tf.layers.BatchNormalization()(conv_out)
                        conv_out += block_input

            conv_out = tf.nn.relu(conv_out)
            conv_out = tf.layers.Flatten()(conv_out)

            conv_out = tf.layers.Dense(256)(conv_out)
            last_layer = tf.nn.relu(conv_out)
            output = tf.layers.Dense(num_outputs)(last_layer)
            return output, last_layer


class BattlesnakeVisionNet(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict["obs"]

        with tf.variable_scope("battlesnake_vision_net"):
            hidden = inputs / 255.0
            hidden = tf.layers.Conv2D(32, 1, 1, activation=tf.nn.leaky_relu)(hidden)
            hidden = tf.layers.Conv2D(64, 2, 2, activation=tf.nn.leaky_relu)(hidden)
            hidden = tf.layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu)(hidden)
            hidden = tf.layers.Flatten()(hidden)
            last_layer = tf.layers.Dense(512, activation=tf.nn.leaky_relu)(hidden)
            output = tf.layers.Dense(num_outputs)(last_layer)
            return output, last_layer


def register():
    ray.tune.register_env("battlesnake", env_creator)
    ModelCatalog.register_custom_model("battlesnake_vision_net", BattlesnakeVisionNet)
    ModelCatalog.register_custom_model("battlesnake_resnet", BattlesnakeResNet)


def on_train_result(info):
    iterations = info["result"]["iterations_since_restore"]
    if iterations % 25 == 0:
        agent = info["agent"]
        agent.export_policy_model(
            os.path.join(agent.logdir, f"model_{iterations}"), "snake_0"
        )


def get_agent_config(
    algorithm: str,
    model: str,
    width: int = 11,
    height: int = 11,
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
    common_config = {
        "env_config": env_config,
        "horizon": 1000,
        "callbacks": {"on_train_result": tune.function(on_train_result)},
    }
    if algorithm == "APEX" or algorithm == "DQN":
        agent_config = {
            "model": {"custom_model": model},
            "num_workers": num_workers,
            "num_envs_per_worker": 32,
            "double_q": False,
            "num_atoms": 51,
            "v_min": -2.0,
            "v_max": env_config["width"] ** 2.0,
            "buffer_size": 1_000_000,
            "exploration_final_eps": 0.01,
            "exploration_fraction": 0.1,
            "prioritized_replay_alpha": 0.5,
            "beta_annealing_fraction": 1.0,
            "final_prioritized_replay_beta": 1.0,
            "multiagent": {
                "policy_graphs": {
                    "snake_0": (DQNPolicyGraph, env.obs_space, env.action_space, {})
                },
                "policy_mapping_fn": tune.function(lambda agent_id: "snake_0"),
            },
        }
    elif algorithm == "IMPALA":
        agent_config = {
            "model": {"custom_model": "battlesnake_vision_net", "use_lstm": True},
            "sample_batch_size": 50,
            "train_batch_size": 500,
            "num_workers": num_workers,
            "num_envs_per_worker": 16,
            "lr_schedule": [[0, 0.0005], [100_000_000, 0.000_000_000_001]],
            "multiagent": {
                "policy_graphs": {
                    "snake_0": (VTracePolicyGraph, env.obs_space, env.action_space, {})
                },
                "policy_mapping_fn": tune.function(lambda agent_id: "snake_0"),
            },
        }
    return {**common_config, **agent_config}


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
    parser.add_argument(
        "--model", help="Network model.", type=str, default="battlesnake_vision_net"
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
                    algorithm=args.algorithm,
                    model=args.model,
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
