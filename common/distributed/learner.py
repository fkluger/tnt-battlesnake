import atexit
import pickle
import zlib
from abc import abstractmethod
from multiprocessing import Process, Queue

import zmq

from common.distributed.server import Server


def run_server_process(
    distributed_config, experience_queue: Queue, parameter_queue: Queue
):
    server = Server(
        distributed_config.learner_parameter_port,
        distributed_config.learner_experience_port,
    )
    experience_buffer = []
    while True:
        socks = dict(server.poller.poll())
        if (
            server.experiences_socket in socks
            and socks[server.experiences_socket] == zmq.POLLIN
        ):
            message = server.experiences_socket.recv_multipart()
            experiences_compressed = message[1]
            experiences_pickled = zlib.decompress(experiences_compressed)
            experiences = pickle.loads(experiences_pickled)
            experience_buffer.extend(experiences)
            if len(experience_buffer) >= 1000:
                experience_queue.put(experience_buffer)
                experience_buffer = []
        if not parameter_queue.empty():
            server.parameter_socket.send_multipart(
                [b"parameters", *parameter_queue.get()]
            )


class Learner:
    def __init__(self, distributed_config):
        self.parameter_queue = Queue()
        self.experience_queue = Queue()

        self.server_process = Process(
            target=run_server_process,
            args=[distributed_config, self.experience_queue, self.parameter_queue],
        )
        self.server_process.start()
        atexit.register(self.kill_server)

    def kill_server(self):
        self.server_process.terminate()

    @abstractmethod
    def _create_parameter_message(self):
        pass

    @abstractmethod
    def _process_experiences(self, experiences):
        pass

    def send_parameters(self):
        self.parameter_queue.put(self._create_parameter_message(), False)

    def receive_experiences(self):
        if self.experience_queue.empty():
            return False
        else:
            experiences = []
            while not self.experience_queue.empty():
                experiences.extend(self.experience_queue.get())
            self._process_experiences(experiences)
            return True
