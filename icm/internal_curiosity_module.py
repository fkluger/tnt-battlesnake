from keras.layers import Input, Conv2D, Flatten, Concatenate, Dense, Subtract, Lambda, Reshape, Multiply, Add
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import plot_model

import tensorflow as tf
import numpy as np

from .feature_encoder import FeatureEncoder
from .forward_model import ForwardModel
from .inverse_model import InverseModel


class ICM:

    def __init__(self, input_shape, num_actions, learning_rate, beta, eta):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.beta = beta
        self.eta = eta
        self.model = self._create_model()

    def compute_internal_rewards(self, observations):
        batch_size = len(observations)
        [states, actions, next_states] = self._create_targets(observations, batch_size)
        internal_rewards = np.zeros([batch_size])
        inverse_losses = np.zeros([batch_size])
        forward_losses = np.zeros([batch_size])
        for idx in range(batch_size):
            prediction = self.model.predict([np.expand_dims(states[idx], 0), np.expand_dims(
                actions[idx], 0), np.expand_dims(next_states[idx], 0)])
            internal_rewards[idx] = prediction[0]
            inverse_losses[idx] = prediction[2]
            forward_losses[idx] = prediction[3]
        return internal_rewards, np.mean(inverse_losses), np.mean(forward_losses)

    def train(self, observations):
        batch_size = len(observations)
        x = self._create_targets(observations, batch_size)
        history = self.model.fit(x=x, batch_size=batch_size, verbose=0)
        return np.mean(history.history['loss'])

    def _create_targets(self, observations, batch_size):
        states = np.zeros((batch_size, ) + self.input_shape, dtype=int)
        actions = np.zeros((batch_size, ), dtype=int)
        next_states = np.zeros((batch_size, ) + self.input_shape, dtype=int)

        for idx, o in enumerate(observations):
            states[idx] = o.state
            actions[idx] = o.action
            if o.next_state is not None:
                next_states[idx] = o.next_state

        return [states, actions, next_states]

    def _create_model(self):
        input_state = Input(shape=self.input_shape, name='state')
        input_next_state = Input(shape=self.input_shape, name='next_state')
        input_action = Input(shape=(1,), name='action')

        features = FeatureEncoder()
        num_features = features.compute_output_shape(self.input_shape)[-1]
        inverse_model = InverseModel(self.num_actions)
        forward_model = ForwardModel(num_features)

        features_state = features(input_state)
        features_next_state = features(input_next_state)

        state_and_next_state = Concatenate(name='state_and_next_state')([features_state, features_next_state])
        action_prediction = inverse_model(state_and_next_state)

        state_and_action = Concatenate(name='state_and_action')([features_state, input_action])
        next_state_prediction = forward_model(state_and_action)

        state_prediction_error = Subtract(name='state_prediction_error')([next_state_prediction, features_next_state])
        internal_reward = Lambda(lambda x: self.eta * 0.5 * tf.reduce_mean(tf.square(x)),
                                 name='internal_reward')(state_prediction_error)

        forward_loss = Lambda(lambda x: self.beta * x, name='forward_loss')(internal_reward)

        inverse_loss = Lambda(lambda x: (1 - self.beta) *
                              tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=tf.to_int32(tf.argmax(input_action, axis=1))), name='inverse_loss')(action_prediction)

        model = Model(inputs=[input_state, input_action, input_next_state],
                      outputs=[internal_reward, action_prediction, inverse_loss, forward_loss])
        model.add_loss([forward_loss, inverse_loss])
        model.compile(optimizer=RMSprop(lr=self.learning_rate))

        # print(model.summary())
        # plot_model(model, to_file='model.png')

        return model
