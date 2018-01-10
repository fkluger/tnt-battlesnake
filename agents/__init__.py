from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def act(self, state):
        '''
        Choose action for the given state.
        ''' 
        pass

    @abstractmethod
    def observe(self, sample):
        '''
        Add sample to memory.
        '''
        pass
    
    @abstractmethod
    def replay(self):
        '''
        Replay samples from memory.
        '''
        pass