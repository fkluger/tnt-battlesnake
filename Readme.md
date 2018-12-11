# tnt-battlesnake

This repository contains a Battlesnake OpenAI Gym environment and Reinforcement Learning Agents, that operate in this environment.

## Quick start

```bash
# Create conda environment
conda create -n bs python=3.6 pip

# Activate environment
conda activate bs

# Install dependencies
pip install sacred gym black pylint tensorflow keras tensorboardX opencv-python gym[atari]

# Train a DQN agent
python run.py train_dqn

# Or run example random agent
python run.py example

# In another terminal window
tensorboard --logdir tmp
```

## Repository structure

- `common`
    - Common models and functions that can be shared between multiple RL algorithms.
- `dqn`
    - Contains a DQN agent with several improvements.
- `experiments`
    - Experiments should have an unique name and a `main.py` file which defines a `sacred` Experiment called `ex` for reproducibility.
- `gym_battlesnake`
    - An OpenAI `gym` environment for Battlesnake.
- `legacy`
    - Contains the old APEX-DQN code that has not been refactored yet.

## Distributed DQN

```bash
# Start Learner
python run.py train_dqn_distributed

# In other terminals
python run.py train_dqn_distributed --actor [number] --skip_observe --learner_address [IP address]

# Or automated via ssh
python start_training.py -l [gpu server] -a [worker server 1] [worker server 2] ... -p [number of actors on worker 1] [number of actors on worker 2]... --gpus [ID of the GPUs]
```