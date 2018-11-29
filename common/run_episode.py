from typing import List

import gym
import numpy as np

from common.models.agent import Agent
from common.models.transition import Transition


def run_episode(environment: gym.Env, agent: Agent, render: bool, max_length: int):
    episode_reward = 0
    episode_length = 0
    state = environment.reset()
    for _ in range(max_length):
        episode_length += 1
        if render:
            environment.render()
        action = agent.act(state)
        next_state, reward, terminal, _ = environment.step(action)
        agent.observe(
            [Transition(state, action, reward, None if terminal else next_state)]
        )
        episode_reward += reward
        if terminal:
            break
        else:
            state = next_state
    return episode_reward, episode_length


def run_episode_vec(
    environments: List[gym.Env], agent: Agent, render: bool, max_length: int
):
    episode_rewards = []
    episode_length = 0
    unfinished_envs = []
    unfinished_envs.extend(environments)
    transition_episodes = {}
    states = [env.reset() for env in unfinished_envs]
    for _ in range(max_length):
        episode_length += 1
        actions = agent.act([state for state in states if state is not None])
        if not actions.shape:
            actions = [actions]

        next_states, rewards, terminals, _ = zip(
            *[env.step(action) for action, env in zip(actions, unfinished_envs)]
        )
        for state, action, reward, next_state, terminal, env in zip(
            states, actions, rewards, next_states, terminals, unfinished_envs
        ):
            if state is not None:
                transition = Transition(
                    state, action, reward, None if terminal else next_state
                )
                if env in transition_episodes:
                    transition_episodes[env].append(transition)
                else:
                    transition_episodes[env] = [transition]
            episode_rewards.append(reward)
        unfinished_envs = [
            env for env, terminal in zip(unfinished_envs, terminals) if not terminal
        ]
        states = next_states
        if not unfinished_envs:
            break
    for episode in transition_episodes.values():
        agent.observe(episode)
    return np.mean(episode_rewards), episode_length
