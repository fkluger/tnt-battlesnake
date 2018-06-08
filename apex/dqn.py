from keras import Model, Input
from keras.layers import Conv2D, Flatten, Lambda, Add, Dense
from keras.optimizers import RMSprop
import tensorflow as tf
import numpy as np


def huber_loss(y_true, y_pred):
    time_difference_error = y_true - y_pred

    cond = tf.abs(time_difference_error) < 1.0
    L2 = 0.5 * tf.square(time_difference_error)
    L1 = 1.0 * (tf.abs(time_difference_error) - 0.5 * 1.0)

    loss = tf.where(cond, L2, L1)

    return tf.reduce_mean(loss)

class DQN:

    def __init__(self, input_shape, num_actions, hidden_size, learning_rate, report_interval):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.model = self._create_model()
        self.callbacks = []

    def _create_model(self):
        inputs = Input(shape=self.input_shape)
        net = Conv2D(32, 6, activation='relu', padding='same')(inputs)
        net = Conv2D(64, 4, activation='relu')(net)
        net = Conv2D(64, 3, activation='relu')(net)
        net = Flatten()(net)
        advt = Dense(512, activation='relu')(net)
        advt = Dense(self.num_actions)(advt)
        value = Dense(512, activation='relu')(net)
        value = Dense(1)(value)
        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keepdims=True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, self.num_actions]))(value)
        final = Add()([value, advt])
        model = Model(inputs=inputs, outputs=final)

        opt = RMSprop(lr=self.learning_rate, decay=0.95, epsilon=1.5e-7)
        model.compile(loss=huber_loss, optimizer=opt)
        return model

    def predict(self, state):
        if len(state.shape) == 3:
            # Single state
            state = np.expand_dims(state, 0)
        return self.model.predict(state)
    
    def compute_q_values(self, observations):
        no_state = np.zeros(self.input_shape)
        next_states = np.array([(no_state
                                 if o.next_state is None else o.next_state)
                                for o in observations])
        states = np.array([o.state for o in observations])
        q_values = np.array(self.model.predict(states))
        q_values_next = np.array(self.model.predict(next_states))
        return q_values, q_values_next
    
    def train(self, x, y, batch_size, weights):
        history = self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            verbose=0,
            sample_weight=weights,
            callbacks=self.callbacks)
        return history.history['loss'][0]
