import time
import logging

from apex.learner import Learner
from apex.configuration import Configuration
from main_utils import wrap_main


def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    learner = Learner(Configuration('./apex/config.json'))
    last_parameter_update = time.time()
    while True:
        learner.update_experiences()
        if time.time() - last_parameter_update > 5:
            last_parameter_update = time.time()
            learner.send_parameters()


if __name__ == '__main__':
    wrap_main(main)
