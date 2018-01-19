from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def get_metrics(self):
        '''
        Get metrics for tensorflow summaries.
        '''
        pass

    @abstractmethod
    def act(self, state):
        '''
        Choose action for the given state.
        '''
        pass

    @abstractmethod
    def observe(self, observation):
        '''
        Add observation to memory.
        '''
        pass

    @abstractmethod
    def replay(self):
        '''
        Replay observations from memory.
        '''
        pass
