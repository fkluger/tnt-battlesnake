from abc import ABC, abstractmethod

class Brain(ABC):

    @abstractmethod
    def predict(self, state):
        '''
        Predict Q-Values for the given state.
        '''
        pass

    @abstractmethod
    def train(self, x, y, batch_size, verbose):
        '''
        Perform supervised training on the sample batch.
        '''
        pass