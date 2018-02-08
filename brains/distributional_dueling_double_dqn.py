import math
from keras import Model, Input
from keras.layers import Conv2D, Flatten, Dense, Lambda, Add, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

from .dueling_double_dqn import DuelingDoubleDQNBrain
from .huber_loss import create_quantile_huber_loss


class DistributionalDuelingDoubleDQNBrain(DuelingDoubleDQNBrain):

    dropout_layers = []
    rate = 1.0
    steps = 0

    def __init__(self, num_quantiles, **kwargs):
        self.num_quantiles = num_quantiles
        self.loss_function = create_quantile_huber_loss(self.num_quantiles)
        super().__init__(**kwargs)

    def create_model(self):
        '''
        Returns a tensor with shape (num_actions, batch_size, num_quantiles).
        '''
        inputs = Input(shape=self.input_shape)
        cnn_features = Conv2D(32, 8, activation='relu', strides=(1, 1))(inputs)
        cnn_features = Conv2D(64, 4, activation='relu', strides=(2, 2))(cnn_features)
        cnn_features = Conv2D(64, 3, activation='relu', strides=(1, 1))(cnn_features)
        cnn_features = Flatten()(cnn_features)
        advantage = Dense(512, activation='relu')(cnn_features)
        advantage = Dropout(self.rate)(advantage)
        self.dropout_layers.append(advantage)
        advantage = Dense(self.num_quantiles * self.num_actions)(advantage)
        value = Dense(512, activation='relu')(cnn_features)
        value = Dropout(self.rate)(value)
        self.dropout_layers.append(value)
        value = Dense(self.num_quantiles)(value)
        # now to combine the two streams
        advantage = Lambda(lambda advantage: advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True))(advantage)
        value = Lambda(lambda value: tf.tile(value, [1, self.num_actions]))(value)
        value_and_advantage = Add()([value, advantage])
        output_distributions = [Dense(self.num_quantiles)(value_and_advantage) for _ in range(self.num_actions)]
        model = Model(inputs=inputs, outputs=output_distributions)

        opt = Adam(lr=self.learning_rate, epsilon=0.01/self.input_shape[0])
        model.compile(loss=self.loss_function, optimizer=opt)
        return model
    
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
