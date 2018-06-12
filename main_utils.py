import logging
import time


def wrap_main(main_fn):
    try:
        main_fn()
    except Exception as e:
        logging.basicConfig(level=logging.DEBUG, filename=f'error-{time.time()}.log',
                            filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.exception(e)
