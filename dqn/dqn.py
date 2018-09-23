import logging

from tensorflow import keras
import numpy as np
import tensorflow as tf

from .huber_loss import huber_loss

LOGGER = logging.getLogger("DQN")


class DQN:
    def __init__(self, input_shape, num_actions, learning_rate):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.online_model = self._create_model()
        print(self.online_model.summary())
        self.target_model = self._create_model()
        self.callbacks = []

    def predict(self, state, target=False):
        state = state.astype(float) / 255.0
        if len(state.shape) == 3:
            # Single state
            state = np.expand_dims(state, 0)
        if target:
            return self.target_model.predict(state)
        else:
            return self.online_model.predict(state)

    def create_targets(self, observations, batch_size):

        q_values, q_values_next, q_values_next_target = self._compute_q_values(
            observations
        )

        x = np.zeros((batch_size,) + self.input_shape)
        y = np.zeros((batch_size, self.num_actions))
        errors = np.zeros(batch_size)

        for idx, o in enumerate(observations):
            target = q_values[idx]
            target_old = np.copy(target)
            clipped_reward = np.clip(o.reward, -1, 1)
            if o.next_state is None:
                target[o.action] = clipped_reward
            else:
                target[o.action] = (
                    clipped_reward
                    + o.discount_factor
                    * q_values_next_target[idx, np.argmax(q_values_next[idx])]
                )
            x[idx] = o.state
            y[idx] = target
            errors[idx] = np.abs(target[o.action] - target_old[o.action])
        return x, y, errors

    def train(self, x, y, batch_size, weights):
        history = self.online_model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            verbose=0,
            sample_weight=weights,
            callbacks=self.callbacks,
        )
        return history.history["loss"][0]

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())
        LOGGER.info("Updated target model.")

    def _compute_q_values(self, observations):
        no_state = np.zeros(self.input_shape)
        next_states = np.array(
            [(no_state if o.next_state is None else o.next_state) for o in observations]
        )
        states = np.array([o.state for o in observations])

        q_values = np.array(self.predict(states))
        q_values_next = np.array(self.predict(next_states))
        q_values_next_target = np.array(self.predict(next_states, target=True))

        return q_values, q_values_next, q_values_next_target

    def _create_model(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        net = keras.layers.Conv2D(32, 3, strides=3, activation="elu")(inputs)
        net = keras.layers.Conv2D(64, 2, strides=2, activation="elu")(net)
        net = keras.layers.Conv2D(64, 1, strides=1, activation="elu")(net)
        net = keras.layers.Flatten()(net)
        advt = keras.layers.Dense(512, activation="elu")(net)
        advt = keras.layers.Dense(self.num_actions)(advt)
        value = keras.layers.Dense(512, activation="elu")(net)
        value = keras.layers.Dense(1)(value)
        advt = keras.layers.Lambda(
            lambda advt: advt - tf.reduce_mean(advt, axis=-1, keepdims=True)
        )(advt)
        value = keras.layers.Lambda(
            lambda value: tf.tile(value, [1, self.num_actions])
        )(value)
        final = keras.layers.Add()([value, advt])
        model = keras.models.Model(inputs=inputs, outputs=final)
        model.compile(
            loss=huber_loss, optimizer=keras.optimizers.RMSprop(lr=self.learning_rate)
        )
        return model
