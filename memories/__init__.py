from abc import ABC, abstractmethod


class Memory(ABC):

    @abstractmethod
    def add(self, observation):
        '''
        Add a sample to the memory.
        '''
        pass

    @abstractmethod
    def sample(self, n):
        '''
        Choose a random sample from the memory.
        '''
        pass
