from abc import ABC, abstractmethod


class Memory(ABC):

    @abstractmethod
    def size(self):
        '''
        Get number of observations in memory.
        '''
        pass

    @abstractmethod
    def add(self, observation):
        '''
        Add a sample to the memory.
        '''
        pass

    @abstractmethod
    def sample(self, n, beta):
        '''
        Choose a random sample from the memory.
        '''
        pass

    @abstractmethod
    def update(self, idx, error):
        '''
        Update an observation with a new time-difference error.
        '''
        pass
