import tensorflow as tf
import ray
import ray.tune as tune
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
import gym

import gym_battlesnake
from gym_battlesnake.envs import BattlesnakeEnv
from gym_battlesnake.wrappers import FrameStack


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
            hidden = tf.layers.Conv2D(16, 1, 1, activation=tf.nn.leaky_relu)(inputs)
            hidden = tf.layers.Conv2D(32, 2, 2, activation=tf.nn.leaky_relu)(hidden)
            hidden = tf.layers.Conv2D(32, 3, 1, activation=tf.nn.leaky_relu)(hidden)
            hidden = tf.layers.Flatten()(hidden)
            last_layer = tf.layers.Dense(256, activation=tf.nn.leaky_relu)(hidden)
            output = tf.layers.Dense(num_outputs)(last_layer)
            return output, last_layer


def main():

    ray.init()
    ray.tune.register_env("battlesnake", env_creator)
    ModelCatalog.register_custom_model("battlesnake_vision_net", BattlesnakeVisionNet)
    env_config = {"width": 7, "height": 7, "num_snakes": 3, "stacked_frames": 2}
    env = env_creator(env_config)
    tune.run_experiments(
        {
            "battlesnake": {
                "run": "DQN",
                "env": "battlesnake",
                "stop": {"episode_reward_mean": 40},
                # "checkpoint_freq": 20,
                "config": {
                    "model": {"custom_model": "battlesnake_vision_net"},
                    "env_config": env_config,
                    "num_workers": 3,
                    "num_envs_per_worker": 32,
                    "num_atoms": 51,
                    "v_min": -2.0,
                    "v_max": 40.0,
                    "noisy": True,
                    "multiagent": {
                        "policy_graphs": {
                            "snake_0": (
                                DQNPolicyGraph,
                                env.obs_space,
                                env.action_space,
                                {},
                            )
                        },
                        "policy_mapping_fn": tune.function(lambda agent_id: "snake_0"),
                    },
                },
            }
        }
    )


if __name__ == "__main__":
    main()
