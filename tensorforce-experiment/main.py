from simulator.environment import BattlesnakeEnvironment
from agents import getAgent
from tensorforce.execution import Runner
import matplotlib.pyplot as plt

def main():
    max_episodes = 1000000
    max_timesteps = max_episodes * 1000

    width = 20
    height = 20
    num_frames = 7

    env = BattlesnakeEnvironment(width, height, 15, 3, num_frames)
    
    agent = getAgent(width, height, num_frames)

    runner = Runner(agent, env)
    
    report_episodes = 100

    def episode_finished(r):
        if r.episode > 0 and r.episode % 1000 == 0:
            env.play_longest_run()
        if r.episode % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            print("Average of last 100 rewards: {:8.4f}".format(sum(r.episode_rewards[-100:]) / 100))
            print("Average episode length: {:8.4f}".format(r.timestep * 1.0 / r.episode))
            print("Longest run: {}".format(env.longest_run))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(episodes=max_episodes, timesteps=max_timesteps, episode_finished=episode_finished)

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    main()