import math
from keras import Model, Input
from keras.layers import Conv2D, Flatten, Dense, Lambda, Add, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
import numpy as np

from .double_dqn import DoubleDQNBrain
from .huber_loss import huber_loss


class DuelingDoubleDQNBrain(DoubleDQNBrain):

    dropout_layers = []
    rate = 1.0
    steps = 0

    def predict(self, state, target=False):
        self.steps += 1
        self.rate = 0.001 + (1.0 - 0.001) * math.exp(-1e-6 * self.steps)
        for layer in self.dropout_layers:
            layer.rate = self.rate
        if self.steps % 10000 == 0:
            print(f'Dropout rate: {self.rate}')
        predictions = []
        for _ in range(3):
            predictions.append(super().predict(state, target))
        return np.mean(predictions, axis=0)

    def create_model(self):
        inputs = Input(shape=self.input_shape)
        net = Conv2D(32, 8, activation='relu', strides=(1, 1))(inputs)
        net = Conv2D(64, 4, activation='relu', strides=(2, 2))(net)
        net = Conv2D(64, 3, activation='relu', strides=(1, 1))(net)
        net = Flatten()(net)
        advt = Dense(512, activation='relu')(net)
        advt = Dropout(self.rate)(advt)
        self.dropout_layers.append(advt)
        advt = Dense(self.num_actions)(advt)
        value = Dense(512, activation='relu')(net)
        value = Dropout(self.rate)(value)
        self.dropout_layers.append(value)
        value = Dense(1)(value)
        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keepdims=True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, self.num_actions]))(value)
        final = Add()([value, advt])
        model = Model(inputs=inputs, outputs=final)

        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss=huber_loss, optimizer=opt)
        return model
