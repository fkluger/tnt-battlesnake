import numpy as np

from . import Runner
from .battlesnake_runner import SimpleRunner


class GymRunner(SimpleRunner):
    '''
    Runner that runs the episodes and collects some statistics.
    '''

    def run(self):
        episodes = len(self.episode_rewards)
        if episodes % 50 == 0:
            print(f'Episode: {episodes}')
        episode_reward = 0
        episode_length = 0
        state = self.simulator.reset()
        terminal = False
        while not terminal:
            self.steps += 1
            self.tensorboard_callback.global_step = self.steps
            action = self.agent.act(state)
            if episodes % 50 == 0:
                self.simulator.render()
            next_state, reward, terminal, _ = self.simulator.step(action)

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
