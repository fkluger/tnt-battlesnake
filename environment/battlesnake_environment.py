from gym import Env

from .state import State
from .constants import Reward
from .environment_statistics import EnvironmentStatistics
from .environment_renderer import EnvironmentRenderer


class BattlesnakeEnvironment(Env):

    def __init__(self, width, height, snakes, fruits, enemy_agents, output_directory):
        self.width = width
        self.height = height
        self.snakes = snakes
        self.fruits = fruits
        self.output_directory = output_directory
        self.stats = EnvironmentStatistics()
        self.renderer = EnvironmentRenderer(output_directory)
        self.enemy_agents = enemy_agents
        self.state = None

        num_agents = len(self.enemy_agents)
        if num_agents != self.snakes - 1:
            raise ValueError(
                f'Need an enemy agent for each enemy snake. Expected {self.snakes - 1} agents, got {num_agents}')

    def reset(self):
        self.state = State(self.width, self.height, self.snakes, self.fruits)
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

        reward, terminal = self._evaluate_reward(fruit_eaten, collided, starved)
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
        self.stats.report()

    def _evaluate_reward(self, fruit_eaten, collided, starved):
        terminal = False
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
