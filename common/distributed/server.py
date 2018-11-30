import atexit
import socket
import zmq


class Server:
    def __init__(self, learner_parameter_port: int, learner_experience_port: int):
        self.ip_address = self._get_ip_address()
        self.learner_parameter_port = learner_parameter_port
        self.learner_experience_port = learner_experience_port
        self._connect_sockets()
        self._add_poller()

    def _get_ip_address(self):
        return socket.gethostbyname(socket.gethostname())

    def _connect_sockets(self):
        self.context = zmq.Context()
        self.parameter_socket = self.context.socket(zmq.PUB)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.bind(
            f"tcp://{self.ip_address}:{self.learner_parameter_port}"
        )

        self.experiences_socket = self.context.socket(zmq.SUB)
        self.experiences_socket.setsockopt(zmq.LINGER, 0)
        self.experiences_socket.setsockopt(zmq.SUBSCRIBE, b"experiences")
        self.experiences_socket.bind(
            f"tcp://{self.ip_address}:{self.learner_experience_port}"
        )
        atexit.register(self._disconnect_sockets)

    def _add_poller(self):
        self.poller = zmq.Poller()
        self.poller.register(self.experiences_socket, zmq.POLLIN)

    def _disconnect_sockets(self):
        self.parameter_socket.close()
        self.experiences_socket.close()
        self.context.term()

