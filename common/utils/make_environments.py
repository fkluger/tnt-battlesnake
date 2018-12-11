import gym
from gym_battlesnake.wrappers import FrameStack


def make_environments(config):
    environments = []
    for _ in range(config.num_envs):
        environment = gym.make(config.env)
        if config.env.startswith("battlesnake"):
            if config.frame_stack > 1:
                environment = FrameStack(
                    environment, num_stacked_frames=config.frame_stack
                )
        environments.append(environment)

    return environments
