import json
import os
from tensorforce.agents import DQNAgent
from simulator.environment import get_state_shape

print(__file__)

def getAgent(width, height, num_frames):
    return DQNAgent(
        states_spec=get_state_shape(width, height, num_frames),
        actions_spec={'type': 'int', 'num_actions': 3},
        network_spec=json.load(open(os.path.join(os.path.dirname(__file__) ,"network.json"))),
        optimizer = {'type': 'adam',
                   'learning_rate': 0.0000625, 'epsilon': 1.5e-4},
        saver_spec = {'seconds': 3000, 'directory': os.path.join(
            os.getcwd(), "model")},
        summary_spec = {'directory': os.path.join(os.getcwd(), "stats"), 'steps': 100,
                      'labels': ['losses']},
        memory = {'type': 'prioritized_replay', 'capacity': 1000000},
        discount = 0.95,
        first_update = 200000,
        variable_noise = 0.5,
        update_frequency = 4,
        repeat_update = 1,
        double_q_model = True,
        target_sync_frequency = 32000,
        target_update_weight = 1.0,
        batch_size = 32,
        states_preprocessing_spec = {
            "type": "divide",
            "scale": 45
        }
    )
