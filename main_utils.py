import logging
import time
import os

LOGGING_DIR = "./logs"


def wrap_main(main_fn):
    try:
        main_fn()
    except Exception as e:
        if not os.path.exists(LOGGING_DIR):
            os.makedirs(LOGGING_DIR)
        logging.root.addHandler(
            logging.FileHandler(filename=f"{LOGGING_DIR}/error-{time.time()}.log")
        )
        logging.exception(e)
