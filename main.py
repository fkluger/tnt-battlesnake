import time
import datetime

from agents.dqn import DQNAgent
from memories.replay import ReplayMemory
from brains.plain_dqn import PlainDQNBrain
from runners.battlesnake_runner import SimpleRunner
from simulator.simulator import BattlesnakeSimulator, get_state_shape


width = 10
height = 10
num_frames = 1
num_snakes = 1
num_fruits = 1
shape = get_state_shape(width, height, num_frames)

simulator = BattlesnakeSimulator(
    width, height, num_snakes, num_fruits, num_frames)

brain = PlainDQNBrain(shape, 3)
memory = ReplayMemory()
agent = DQNAgent(brain, memory, shape, 3)
runner = SimpleRunner(agent, simulator)

episodes = 0
max_episodes = 200e6
report_interval = 10

while episodes < max_episodes:
    episodes += 1
    runner.run()

    if episodes % report_interval == 0:
        mean_episode_length = sum(runner.episode_lengths[-report_interval:]) * 1.0 / report_interval
        mean_episode_rewards = sum(runner.episode_rewards[-report_interval:]) * 1.0 / report_interval
        ts = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y %H:%M:%S')
        print('{} - Episode: {}\tMean reward: {:4.4f}\tMean length: {:4.4f}'.format(ts, episodes, mean_episode_rewards, mean_episode_length))
    if episodes % (report_interval * 10) == 0:
        simulator.play_longest_episode()

