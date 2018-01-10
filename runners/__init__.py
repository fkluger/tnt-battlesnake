from abc import ABC, abstractmethod

class Runner(ABC):

    @abstractmethod
    def run(self):
        '''
        Runs one episode.
        '''
        pass