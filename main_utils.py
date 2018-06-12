import logging
import sys
import time
import traceback


def extract_function_name():
    """Extracts failing function name from Traceback
    by Alex Martelli
    http://stackoverflow.com/questions/2380073/\
    how-to-identify-what-function-call-raise-an-exception-in-python
    """
    tb = sys.exc_info()[-1]
    stk = traceback.extract_tb(tb, 1)
    fname = stk[0][3]
    return fname


def log_exception(e):
    logging.error(f'Function {extract_function_name()} raised {e.__class__} ({e.__doc__}): {e.message}')


def wrap_main(main_fn):
    try:
        main_fn()
    except Exception as e:
        logging.basicConfig(level=logging.DEBUG, filename=f'error-{time.time()}.log',
                            filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_exception(e)
