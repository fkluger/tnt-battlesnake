from keras import Sequential, Input
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import RMSprop

from . import Brain


class PlainDQNBrain(Brain):
    '''
    Brain that encapsulates the DQN CNN.
    '''

    def __init__(self, input_shape, num_actions, learning_rate=0.00025):
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=input_shape))
        self.model.add(Conv2D(32, (8, 8), activation='relu', padding='same'))
        self.model.add(Conv2D(64, (4, 4), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=num_actions, activation='linear'))

        opt = RMSprop(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=opt)

    def predict(self, state):
        return self.model.predict(state)

    def train(self, x, y, batch_size, verbose=0):
        return self.model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose)
