from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import Callback

from . import Brain

class PlainDQNBrain(Brain):
    '''
    Brain that encapsulates the DQN CNN.
    '''

    def __init__(self, input_shape, num_actions, learning_rate=0.00025):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.input_shape))
        model.add(Conv2D(32, (8, 8), activation='relu', padding='same'))
        model.add(Conv2D(64, (4, 4), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.num_actions, activation='linear'))

        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        return model

    def update_target(self):
        pass

    def predict(self, state, target=False):
        return self.model.predict(state)

    def train(self, x, y, batch_size, weights):
        return self.model.fit(x=x, y=y, batch_size=batch_size, verbose=0, sample_weight=weights)
