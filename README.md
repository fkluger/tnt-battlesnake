# tnt-battlesnake

## Usage

```bash
# install requirements
pip install -r requirements.txt

# train agent
python main.py

# start battlesnake server at port 8080
python snake_server.py 8080 2 ./experiment - */*.h5
```

## Implemented Agents

- DQN with epsilon-greedy exploration strategy

## Implemented Extensions

- Experience replay
- Prioritized experience replay
- Double DQN
- Dueling Double DQN