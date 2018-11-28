import time

from dqn import Learner
from apex import Configuration
from main_utils import wrap_main


def main():
    config = Configuration("./apex/config.json")
    learner = Learner(config)
    last_parameter_update = time.time()
    while True:
        learner.update_experiences()
        learner.evaluate_experiences()
        if time.time() - last_parameter_update > config.parameter_update_interval / 10.0:
            last_parameter_update = time.time()
            learner.send_parameters()


if __name__ == "__main__":
    wrap_main(main)
