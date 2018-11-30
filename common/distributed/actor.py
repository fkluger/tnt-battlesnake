from typing import List

import pickle
import zmq
import zlib

from common.distributed.client import Client
from common.models.transition import Transition


class Actor:
    def __init__(self, distributed_config):
        self.client = Client(
            distributed_config.learner_address,
            distributed_config.learner_parameter_port,
            distributed_config.learner_experience_port,
        )
        self.buffer: List[Transition] = []
        self.max_buffer_size = distributed_config.max_buffer_size

    def send_experiences(self):
        experiences_pickled = pickle.dumps(self.buffer)
        experiences_compressed = zlib.compress(experiences_pickled)
        self.client.experience_socket.send_multipart(
            [b"experiences", experiences_compressed]
        )
        self.buffer.clear()

    def parameters_received(self):
        try:
            return self.client.parameter_socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
