import atexit
import zmq


class Client:
    def __init__(
        self,
        learner_address: str,
        learner_parameter_port: int,
        learner_experience_port: int,
    ):
        self.learner_address = learner_address
        self.learner_parameter_port = learner_parameter_port
        self.learner_experience_port = learner_experience_port
        self._connect_sockets()

    def _connect_sockets(self):
        self.context = zmq.Context()
        self.parameter_socket = self.context.socket(zmq.SUB)
        self.parameter_socket.setsockopt(zmq.LINGER, 0)
        self.parameter_socket.connect(
            f"tcp://{self.learner_address}:{self.learner_parameter_port}"
        )
        self.parameter_socket.setsockopt(zmq.SUBSCRIBE, b"parameters")

        self.experience_socket = self.context.socket(zmq.PUB)
        self.experience_socket.setsockopt(zmq.LINGER, 0)
        self.experience_socket.connect(
            f"tcp://{self.learner_address}:{self.learner_experience_port}"
        )
        atexit.register(self._disconnect_sockets)

    def _disconnect_sockets(self):
        self.experience_socket.close()
        self.parameter_socket.close()
        self.context.term()
