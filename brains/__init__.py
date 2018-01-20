from abc import ABC, abstractmethod


class Brain(ABC):

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def predict(self, state, target=False):
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
