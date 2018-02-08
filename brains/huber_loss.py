import tensorflow as tf
import numpy as np
import math

HUBER_LOSS_DELTA = 1.0


def huber_loss(y_true, y_pred):
    time_difference_error = y_true - y_pred

    cond = tf.abs(time_difference_error) < HUBER_LOSS_DELTA
    L2 = 0.5 * tf.square(time_difference_error)
    L1 = HUBER_LOSS_DELTA * (tf.abs(time_difference_error) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)

    return tf.reduce_mean(loss)


def create_quantile_huber_loss(num_quantiles):
    tau = tf.range(1.0 / num_quantiles, 1.0 + 1.0 / num_quantiles, 1.0 / num_quantiles)

    def quantile_huber_loss(y_true, y_pred):
        loss = huber_loss(y_true, y_pred)
        time_difference_error = y_true - y_pred
        delta = tf.cast(time_difference_error < 0, tf.float32)
        return tf.reduce_mean(tf.abs(tau - delta) * loss)
    return quantile_huber_loss


def np_huber_loss(y_true, y_pred):
    time_difference_error = y_true - y_pred

    cond = abs(time_difference_error) < HUBER_LOSS_DELTA
    L2 = 0.5 * time_difference_error**2
    L1 = HUBER_LOSS_DELTA * (abs(time_difference_error) - 0.5 * HUBER_LOSS_DELTA)

    loss = np.where(cond, L2, L1)

    return np.mean(loss)


def create_np_quantile_huber_loss(num_quantiles):
    tau = np.arange(1.0 / num_quantiles, 1.0 + 1.0 / num_quantiles, 1.0 / num_quantiles)

    def quantile_huber_loss(y_true, y_pred):
        loss = np_huber_loss(y_true, y_pred)
        time_difference_error = y_true - y_pred
        delta = [time_difference_error < 0]
        return np.mean(abs(tau - delta) * loss)
    return quantile_huber_loss
