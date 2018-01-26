from keras import Model, Input
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Lambda, Add
from keras.optimizers import RMSprop
import tensorflow as tf

from .double_dqn import DoubleDQNBrain
from .huber_loss import huber_loss



class DuelingDoubleDQNBrain(DoubleDQNBrain):

    def create_model(self):
        inputs = Input(shape=self.input_shape)
        net = Conv2D(32, 8, activation='relu', padding='same')(inputs)
        net = Conv2D(64, 4, activation='relu')(net)
        net = Conv2D(64, 3, activation='relu')(net)
        net = Flatten()(net)
        advt = Dense(256, activation='relu')(net)
        advt = Dense(self.num_actions)(advt)
        value = Dense(256, activation='relu')(net)
        value = Dense(1)(value)
        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keep_dims=True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, self.num_actions]))(value)
        final = Add()([value, advt])
        model = Model(inputs=inputs, outputs=final)

        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss=huber_loss, optimizer=opt)
        return model
