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

    return loss


def create_quantile_huber_loss(num_quantiles):
    tau_i = tf.range(num_quantiles + 1, dtype=tf.float32) / num_quantiles * 1.0

    def quantile_huber_loss(y_true, y_pred):
        tau_hat_i = 0.5 * (tau_i[1:] + tau_i[:-1]) # quantile midpoints
        error = tf.cast(y_true - y_pred, tf.float32)
        loss = huber_loss(y_true, y_pred)
        delta = tf.cast(error < 0, tf.float32)
        quantile_loss = tf.abs(tau_hat_i - delta) * loss
        return tf.reduce_mean(quantile_loss)
    return quantile_huber_loss


def np_huber_loss(y_true, y_pred):
    time_difference_error = y_true - y_pred

    cond = abs(time_difference_error) < HUBER_LOSS_DELTA
    L2 = 0.5 * time_difference_error**2
    L1 = HUBER_LOSS_DELTA * (abs(time_difference_error) - 0.5 * HUBER_LOSS_DELTA)

    loss = np.where(cond, L2, L1)

    return loss


def create_np_quantile_huber_loss(num_quantiles):
    tau = np.arange(num_quantiles + 1) / float(num_quantiles)

    def quantile_huber_loss(y_true, y_pred):
        tau_hat = 0.5 * (tau[1:] + tau[:-1])
        error = y_true - y_pred
        loss = np_huber_loss(y_true, y_pred)
        delta = [error < 0]
        quantile_loss = abs(tau_hat - delta) * loss
        return np.mean(quantile_loss)
    return quantile_huber_loss
