import tensorflow as tf
import ray
import ray.tune as tune
from ray.rllib.models import Model, ModelCatalog
import gym

import gym_battlesnake
from gym_battlesnake.wrappers import FrameStack


def env_creator(config):
    env = gym.make(config["name"])
    if config["frame_stack"] > 1:
        env = FrameStack(env, num_stacked_frames=config["frame_stack"])
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
    tune.run_experiments(
        {
            "battlesnake": {
                "run": "DQN",
                "env": "battlesnake",
                "stop": {"episode_reward_mean": 40},
                "config": {
                    "model": {"custom_model": "battlesnake_vision_net"},
                    "env_config": {"frame_stack": 2, "name": "battlesnake-v0"},
                    "num_envs_per_worker": 32,
                },
            }
        }
    )


if __name__ == "__main__":
    main()
