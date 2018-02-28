import numpy as np

from . import Runner


class SimpleRunner(Runner):
    '''
    Runner that runs the episodes and collects some statistics.
    '''

    episode_rewards = []
    episode_lengths = []
    q_value_estimates = []

    steps = 0

    def __init__(self, agent, simulator, training_interval, report_interval, tensorboard_callback):
        self.agent = agent
        self.simulator = simulator
        self.training_interval = training_interval
        self.report_interval = report_interval
        self.tensorboard_callback = tensorboard_callback

    def get_metrics(self):
        mean_episode_length = sum(self.episode_lengths[
            -self.report_interval:]) * 1.0 / self.report_interval
        mean_episode_rewards = sum(self.episode_rewards[
            -self.report_interval:]) * 1.0 / self.report_interval
        
        return [{
            'name': 'runner/mean rewards',
            'value': mean_episode_rewards,
            'type': 'value'
        }, {
            'name': 'runner/mean episode length',
            'value': mean_episode_length,
            'type': 'value'
        }]

    def run(self):
        episode_reward = 0
        episode_length = 0
        state = self.simulator.reset()
        terminal = False
        while not terminal:
            self.steps += 1
            self.tensorboard_callback.global_step = self.steps
            action = self.agent.act(state)
            next_state, reward, terminal = self.simulator.step([action])

            episode_reward += reward
            episode_length += 1

            if terminal:
                next_state = None

            self.agent.observe((state, action, reward, next_state))
            if hasattr(self.agent,
                       'brain') and self.steps % self.training_interval == 0:
                self.agent.replay()
            state = next_state

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
