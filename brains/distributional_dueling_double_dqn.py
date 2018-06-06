from keras import Model, Input
from keras.layers import Conv2D, Flatten, Lambda, Add, Reshape
from keras.optimizers import Adam
import tensorflow as tf

from brains.double_dqn import DoubleDQNBrain
from .huber_loss import create_quantile_huber_loss
from .noisy_dense_layer import NoisyDense


class DistributionalDuelingDoubleDQNBrain(DoubleDQNBrain):

    def __init__(self, num_quantiles, **kwargs):
        self.num_quantiles = num_quantiles
        self.hidden_size = 256
        self.loss_function = create_quantile_huber_loss(self.num_quantiles)
        super().__init__(**kwargs)

    def create_model(self):
        '''
        Returns a tensor with shape (batch_size, num_actions, num_quantiles).
        '''
        inputs = Input(shape=self.input_shape)
        is_image_input = len(self.input_shape) > 1
        if is_image_input:
            cnn_features = Conv2D(32, 3, activation='relu', strides=(1, 1))(inputs)
            cnn_features = Conv2D(64, 2, activation='relu', strides=(2, 2))(cnn_features)
            cnn_features = Conv2D(64, 2, activation='relu', strides=(1, 1))(cnn_features)
            cnn_features = Flatten()(cnn_features)
            advt = NoisyDense(self.hidden_size, activation='relu')(cnn_features)
        else:
            advt = NoisyDense(self.hidden_size, activation='relu')(inputs)
        advt = NoisyDense(self.num_quantiles * self.num_actions)(advt)
        advt = Reshape((self.num_actions, self.num_quantiles))(advt)
        if is_image_input:
            value = NoisyDense(self.hidden_size, activation='relu')(cnn_features)
        else:
            value = NoisyDense(self.hidden_size, activation='relu')(inputs)
        value = NoisyDense(self.num_quantiles)(value)
        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-2, keepdims=True))(advt)
        value = Lambda(lambda value: tf.tile(tf.expand_dims(value, -2), [1, self.num_actions, 1]))(value)
        quantile_output = Add()([value, advt])
        model = Model(inputs=inputs, outputs=quantile_output)

        opt = Adam(lr=self.learning_rate, epsilon=0.01/self.input_shape[0])
        model.compile(loss=self.loss_function, optimizer=opt)
        return model
    
