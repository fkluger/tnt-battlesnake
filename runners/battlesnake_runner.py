import numpy as np

from . import Runner


class SimpleRunner(Runner):
    '''
    Runner that runs the episodes and collects some statistics.
    '''

    episode_rewards = []
    episode_lengths = []
    losses = []
    q_value_estimates = []

    def __init__(self, agent, simulator):
        self.agent = agent
        self.simulator = simulator

    def run(self):
        episode_reward = 0
        episode_length = 0
        episode_q_value_estimates = []
        episode_losses = []
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
            loss, mean_q_estimates = self.agent.replay()
            episode_q_value_estimates.append(mean_q_estimates)
            episode_losses.append(loss)
            state = next_state
        self.losses.append(np.mean(episode_losses))
        self.q_value_estimates.append(np.mean(episode_q_value_estimates))
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

