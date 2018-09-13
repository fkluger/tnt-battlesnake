from typing import Dict
import json


class Configuration:
    def __init__(self, path):
        with open(path) as f:
            config = json.load(f)
            self._config = config

            self.report_interval: int = config["report_interval"]
            self.render_interval: int = config["render_interval"]
            self.output_directory: str = config["output_directory"]

            self.num_actions: int = config["num_actions"]
            self.width: int = config["width"]
            self.height: int = config["height"]
            self.snakes: int = config["snakes"]
            self.fruits: int = config["fruits"]
            self.sparse_rewards: bool = config["sparse_rewards"]

            self.random_initial_steps: int = config["random_initial_steps"]
            self.batch_size: int = config["batch_size"]
            self.learning_rate: float = config["learning_rate"]
            self.discount_factor: float = config["discount_factor"]
            self.target_update_interval: int = config["target_update_interval"]
            self.training_interval: int = config["training_interval"]
            self.parameter_update_interval: int = config["parameter_update_interval"]
            self.multi_step_n: int = config["multi_step_n"]
            self.stacked_frames: int = config["stacked_frames"]

            self.epsilon_base: float = config["epsilon_base"]

            self.replay_capacity: int = config["replay_capacity"]
            self.replay_min_priority: float = config["replay_min_priority"]
            self.replay_max_priority: float = config["replay_max_priority"]
            self.replay_prioritization_factor: float = config[
                "replay_prioritization_factor"
            ]
            self.replay_importance_weight: float = config["replay_importance_weight"]
            self.replay_importance_weight_annealing_step_size: float = config[
                "replay_importance_weight_annealing_step_size"
            ]

            self.learner_ip_address: str = config["learner_ip_address"]
            self.starting_port: str = config["starting_port"]
            self.actors: Dict[str, int] = config["actors"]
            self.actor_buffer_size: int = config["actor_buffer_size"]

            self.nec_key_length: int = config["nec_key_length"]
            self.nec_capacity: int = config["nec_capacity"]
            self.nec_num_nearest_neighbours: int = config["nec_num_nearest_neighbours"]
            self.nec_delta: float = config["nec_delta"]
            self.nec_learning_rate: float = config["nec_learning_rate"]

    def get_input_shape(self):
        return [self.width, self.height, self.stacked_frames]

    def get_num_actors(self) -> int:
        num_actors = 0
        actor_ip_addresses = self.actors.keys()
        for ip_address in actor_ip_addresses:
            num_actors += self.actors[ip_address]
        return num_actors
