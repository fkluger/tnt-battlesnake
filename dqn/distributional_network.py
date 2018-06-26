import logging
import math

from keras import Model, Input
from keras.layers import Conv2D, Flatten, Dense, Reshape, Softmax
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import numpy as np

from .network import DQN

LOGGER = logging.getLogger('DistributionalDQN')


def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -(1.0/m) * np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))


class DistributionalDQN(DQN):

    def __init__(self, num_atoms: int, v_max: float, v_min: float, **kwargs):
        self.num_atoms = num_atoms
        self.v_max = v_max
        self.v_min = v_min
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = np.array([self.v_min + i * self.delta_z for i in range(self.num_atoms)])
        super().__init__(**kwargs)

    def predict(self, state, target=False):
        z = super().predict(state, target)
        # Shape of z is (batch_size, num_actions, num_atoms), so this returns (batch_size, num_actions)
        return np.sum(z * self.z, axis=-1)

    def create_targets(self, observations, batch_size):
        z, z_next, z_next_target = self._compute_q_values(observations)

        x = np.zeros((batch_size, ) + self.input_shape)
        y = np.zeros((batch_size, self.num_actions, self.num_atoms))
        errors = np.zeros(batch_size)

        best_actions = np.argmax(np.sum(z_next * self.z, axis=-1), axis=-1)

        for idx, o in enumerate(observations):
            target = np.zeros((self.num_actions, self.num_atoms))
            target_old = np.copy(z[idx])
            if o.next_state is None:
                # Clip reward into support
                Tz = np.clip(o.reward, self.v_min, self.v_max)
                # Find position in support
                bj = (Tz - self.v_min) / self.delta_z
                # Distribute reward between the 2 atoms above and below the position
                m_l, m_u = math.floor(bj), math.ceil(bj)
                target[o.action, int(m_l)] += (m_u - bj)
                target[o.action, int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    # Clip reward into support
                    Tz = np.clip(o.reward + o.discount_factor * self.z[j], self.v_min, self.v_max)
                    # Find position in support
                    bj = (Tz - self.v_min) / self.delta_z
                    # Distribute reward between the 2 atoms above and below the position
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    target[o.action, int(m_l)] += z_next_target[idx, best_actions[idx], j] * (m_u - bj)
                    target[o.action, int(m_u)] += z_next_target[idx, best_actions[idx], j] * (bj - m_l)
            x[idx] = o.state
            y[idx] = target
            errors[idx] = cross_entropy(target[o.action], target_old[o.action])
        return x, y, errors

    def _create_model(self):
        inputs = Input(shape=self.input_shape)
        net = Conv2D(32, 1, strides=1, activation='relu')(inputs)
        net = Conv2D(64, 2, strides=2, activation='relu')(net)
        net = Conv2D(64, 4, strides=1, activation='relu')(net)
        net = Flatten()(net)
        net = Dense(512, activation='relu')(net)
        net = Dense(self.num_actions * self.num_atoms)(net)
        net = Reshape((self.num_actions, self.num_atoms))(net)
        net = Softmax()(net)
        model = Model(inputs=inputs, outputs=net)

        model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=self.learning_rate, amsgrad=True))
        return model
