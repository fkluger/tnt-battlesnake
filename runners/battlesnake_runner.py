import tensorflow as tf

from . import Runner


class SimpleRunner(Runner):
    '''
    Runner that runs the episodes and collects some statistics.
    '''

    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_q_value_estimates = []

    def __init__(self, agent, simulator):
        self.agent = agent
        self.simulator = simulator

    def run(self):
        episode_reward = 0
        episode_length = 0
        state = self.simulator.reset()
        terminal = False
        while not terminal:
            action = self.agent.act(state)
            next_state, reward, terminal = self.simulator.step([action])

            episode_reward += reward
            episode_length += 1

            if terminal:
                next_state = None

            self.agent.observe((state, action, reward, next_state))
            self.agent.replay()
            state = next_state
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
