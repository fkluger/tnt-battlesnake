import logging
import socket
from contextlib import closing

LOGGER = logging.getLogger('ApexDQN')


def get_free_port(starting_port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        for i in range(200):
            port = starting_port + i
            try:
                s.bind(('', port))
                return s.getsockname()[1]
            except Exception:
                LOGGER.debug(
                    f'Port {port} is already in use. Trying next port...')


def get_ip_address():
    return socket.gethostbyname(socket.gethostname())
