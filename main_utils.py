import logging
import time


def wrap_main(main_fn):
    try:
        main_fn()
    except Exception as e:
        logging.root.addHandler(logging.FileHandler(filename=f'error-{time.time()}.log'))
        logging.exception(e)
