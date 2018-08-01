from gym import Env

from .state import State
from .constants import Reward
from .environment_statistics import EnvironmentStatistics
from .environment_renderer import EnvironmentRenderer


class BattlesnakeEnvironment(Env):

    def __init__(self, config, enemy_agents, output_directory, actor_idx, tensorboard_logger):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.snakes = config.snakes
        self.fruits = config.fruits
        self.stacked_frames = config.stacked_frames
        self.output_directory = output_directory
        self.stats = EnvironmentStatistics(tensorboard_logger, actor_idx)
        self.renderer = EnvironmentRenderer(output_directory)
        self.enemy_agents = enemy_agents
        self.state = None

        num_agents = len(self.enemy_agents)
        if num_agents != self.snakes - 1:
            raise ValueError(
                f'Need an enemy agent for each enemy snake. Expected {self.snakes - 1} agents, got {num_agents}')

    def reset(self):
        self.state = State(self.width, self.height, self.stacked_frames, self.snakes, self.fruits)
        self.stats.on_reset()
        self.renderer.on_reset()
        state = self.state.observe()
        self.renderer.add_frame(state)
        return state

    def step(self, action):

        actions = [action]

        for idx, enemy in enumerate(self.enemy_agents):
            # Index 0 is the agent snake
            enemy_action = enemy.act(self.state, idx + 1)
            actions.append(enemy_action)

        fruit_eaten, collided, starved, won = self.state.move_snakes(actions)

        reward, terminal = self._evaluate_reward(fruit_eaten, collided, starved, won)
        terminal = terminal or won

        if terminal:
            next_state = None
        else:
            next_state = self.state.observe()
            self.renderer.add_frame(next_state)

        self.stats.on_step(fruit_eaten, reward)

        return next_state, reward, terminal

    def render(self, mode='human'):
        episode = self.stats.episodes
        steps = self.stats.episode_steps[-1]
        fruits = self.stats.episode_fruits[-1]
        self.renderer.render(f'episode-{episode}-steps-{steps}-fruits-{fruits}.mp4')

    def _evaluate_reward(self, fruit_eaten, collided, starved, won):
        terminal = False
        if self.config.sparse_rewards:
            reward = 0
            if collided or starved:
                reward = Reward.lost
                terminal = True
            elif won:
                reward = Reward.won
                terminal = True
        else:
            reward = Reward.nothing
            if collided:
                terminal = True
                reward = Reward.collision
            else:
                if fruit_eaten:
                    reward = Reward.fruit
                elif starved:
                    terminal = True
                    reward = Reward.starve
        return reward, terminal
