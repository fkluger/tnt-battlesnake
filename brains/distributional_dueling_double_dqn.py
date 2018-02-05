from keras import Model, Input
from keras.layers import Conv2D, Flatten, Dense, Lambda, Add
from keras.optimizers import Adam
import tensorflow as tf

from .dueling_double_dqn import DuelingDoubleDQNBrain
from .huber_loss import create_quantile_huber_loss


class DistributionalDuelingDoubleDQNBrain(DuelingDoubleDQNBrain):

    def __init__(self, num_quantiles, **kwargs):
        self.num_quantiles = num_quantiles
        self.loss_function = create_quantile_huber_loss(self.num_quantiles)
        super().__init__(**kwargs)

    def create_model(self):
        '''
        Returns a tensor with shape (num_actions, batch_size, num_quantiles).
        '''
        inputs = Input(shape=self.input_shape)
        cnn_features = Conv2D(32, 8, activation='relu', padding='same')(inputs)
        cnn_features = Conv2D(64, 4, activation='relu')(cnn_features)
        cnn_features = Conv2D(64, 3, activation='relu')(cnn_features)
        cnn_features = Flatten()(cnn_features)
        advantage = Dense(256, activation='relu')(cnn_features)
        advantage = Dense(self.num_quantiles * self.num_actions)(advantage)
        value = Dense(256, activation='relu')(cnn_features)
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
