import tensorflow as tf
import numpy as np

from keras.layers import Dense
from keras.engine import InputSpec


class NoisyDense(Dense):

    def get_metrics(self):
        return [
            tf.summary.scalar('noisy_layers/' + self.name + '_mean_w_sigma', self.mean_w_sigma),
            tf.summary.scalar('noisy_layers/' + self.name + '_mean_b_sigma', self.mean_b_sigma)
        ]

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        # Initializer for factorized gaussian noise of \mu and \sigma (see section 3.2)
        sigma_0 = 0.5
        mu_init = tf.random_uniform_initializer(
            minval=-1.0 / np.sqrt(input_dim), maxval=1.0 / np.sqrt(input_dim))
        sigma_init = tf.constant_initializer(sigma_0 / np.sqrt(input_dim))

        # See section 3b
        def f(x):
            return tf.sign(x) * tf.sqrt(tf.abs(x))

        # Sample noise from gaussian
        p = tf.random_normal([input_dim, 1])
        q = tf.random_normal([1, self.units])
        f_p = f(p)
        f_q = f(q)
        w_epsilon = f_p * f_q
        b_epsilon = tf.squeeze(f_q)

        w_mu = self.add_weight(
            name=self.name + 'w_mu',
            shape=[input_dim, self.units],
            initializer=mu_init)
        w_sigma = self.add_weight(
            name=self.name + 'w_sigma',
            shape=[input_dim, self.units],
            initializer=sigma_init)

        b_mu = self.add_weight(
            name=self.name + 'b_mu', shape=[self.units], initializer=mu_init)
        b_sigma = self.add_weight(
            name=self.name + 'b_sigma',
            shape=[self.units],
            initializer=sigma_init)

        self.mean_w_sigma = tf.reduce_mean(w_sigma, name=self.name + 'mean_w_sigma')
        self.mean_b_sigma = tf.reduce_mean(b_sigma, name=self.name + 'mean_b_sigma')

        self.kernel = w_mu + tf.multiply(w_sigma, w_epsilon)
        if self.use_bias:
            self.bias = b_mu + tf.multiply(b_sigma, b_epsilon)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
