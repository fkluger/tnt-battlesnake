# tnt-battlesnake

## Usage

```bash
# install requirements
pip install -r requirements.txt

# train agent on battlesnake
python main.py

# train agent on gym
python main.py --gym_env CartPole-v1 --random_steps 1000
```

## Implemented Agents

- DQN with NoisyNets for exploration

## Implemented Extensions

- Experience replay
- Prioritized experience replay
- Double DQN
- Dueling Double DQN