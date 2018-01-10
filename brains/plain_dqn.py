from keras import Sequential, Input
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from . import Brain


class PlainDQNBrain(Brain):

    tensorboard_callback = TensorBoard(log_dir='./logs')

    def __init__(self, input_shape, num_actions, learning_rate=0.00025, verbose=False):

        self.input_shape = input_shape

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
        if verbose:
            self.tensorboard_callback.set_model(self.model)
            


    def predict(self, state):
        return self.model.predict(state)

    def train(self, x, y, batch_size, verbose=0):
        self.model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose)
