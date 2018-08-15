import atexit
import logging
import time

import tensorflow as tf

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True

import keras.backend as K

K.set_session(tf.Session(config=session_config))


def close_session():
    K.get_session().close()


atexit.register(close_session)

from apex.learner import Learner
from apex.configuration import Configuration
from main_utils import wrap_main


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    learner = Learner(Configuration("./apex/config.json"))
    last_parameter_update = time.time()
    while True:
        learner.update_experiences()
        if time.time() - last_parameter_update > 5:
            last_parameter_update = time.time()
            learner.send_parameters()


if __name__ == "__main__":
    wrap_main(main)
