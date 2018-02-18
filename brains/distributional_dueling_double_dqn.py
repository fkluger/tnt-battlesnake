import math
from keras import Model, Input
from keras.layers import Conv2D, Flatten, Dense, Lambda, Add, Dropout, Reshape
from keras.optimizers import Adam
import keras.backend as K
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
        Returns a tensor with shape (batch_size, num_actions, num_quantiles).
        '''
        inputs = Input(shape=self.input_shape)
        cnn_features = Conv2D(32, 3, activation='relu', strides=(1, 1))(inputs)
        cnn_features = Conv2D(64, 5, activation='relu', strides=(2, 2))(cnn_features)
        cnn_features = Conv2D(64, 3, activation='relu', strides=(1, 1))(cnn_features)
        cnn_features = Flatten()(cnn_features)
        hidden = Dense(512, activation='relu')(cnn_features)
        hidden = Dropout(self.rate)(hidden)
        self.dropout_layers.append(hidden)
        quantile_output = Dense(self.num_quantiles * self.num_actions)(hidden)
        quantile_output = Reshape((self.num_actions, self.num_quantiles))(quantile_output)
        model = Model(inputs=inputs, outputs=quantile_output)

        opt = Adam(lr=self.learning_rate, epsilon=0.01/self.input_shape[0])
        model.compile(loss=self.loss_function, optimizer=opt)
        return model
    
