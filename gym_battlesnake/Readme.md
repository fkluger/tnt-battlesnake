# gym-battlesnake

An [OpenAI Gym](https://gym.openai.com/) environment modeled after the game [Battlesnake](https://www.battlesnake.io/). The agent has to eat the fruits without colliding with itself, other snakes or the walls. Additionally the agent starves if it has not eaten a fruit for 100 steps.

## Usage

```python
import gym
import gym_battlesnake
from gym_battlesnake.wrappers import FrameStack

...

agent = ... # Some RL agent
environment = FrameStack(gym.make("battlesnake-15x15-hard-v0"), num_stacked_frames=2)
for episode in range(episodes):
    state = environment.reset()
    for _ in range(max_episode_length):
        environment.render()
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.observe(state, action, reward, None if done else next_state)
        if done:
            break
        else:
            state = next_state
```

## Variants

### Board size, number of fruits and sparse rewards

```python
environment = gym.make("battlesnake-[width]x[height]-[easy|medium|hard]-<sparse>-v0")
```

Available board sizes are:
- 9
- 12
- 15
- 18

### Number of enemy snakes

Use the `Multiplayer` wrapper to add enemy snakes to the environment.

```python
import gym
import gym_battlesnake
from gym_battlesnake.wrappers import Multiplayer

...

enemy_agents = [...] # Some classes implementing the `Agent` interface
environment = Multiplayer(gym.make("battlesnake-15x15-hard-v0"), enemy_agents=enemy_agents)

...
```
