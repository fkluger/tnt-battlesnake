from abc import ABC, abstractmethod

class Memory(ABC):

    @abstractmethod
    def add(self, sample):
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
