import gym

from common.models.agent import Agent
from common.models.transition import Transition


def run_episode(environment: gym.Env, agent: Agent, render: bool, max_length: int):
    episode_reward = 0
    state = environment.reset()
    for _ in range(max_length):
        if render:
            environment.render()
        action = agent.act(state)
        next_state, reward, terminal, _ = environment.step(action)
        agent.observe(
            Transition(state, action, reward, None if terminal else next_state)
        )
        episode_reward += reward
        if terminal:
            break
        else:
            state = next_state
    return episode_reward
