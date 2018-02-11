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
    tau = tf.range(num_quantiles + 1, dtype=tf.float32) / num_quantiles * 1.0

    def quantile_huber_loss(y_true, y_pred):
        tau_hat = 0.5 * (tau[1:] + tau[:-1])
        sorted_quantiles = tf.nn.top_k(y_pred[1, :], k=num_quantiles, sorted=True).indices[0]
        tau_hat = tau_hat[sorted_quantiles]
        time_difference_error = tf.cast(y_true - y_pred, tf.float32)
        cond = tf.abs(time_difference_error) < HUBER_LOSS_DELTA
        L2 = 0.5 * tf.square(time_difference_error)
        L1 = HUBER_LOSS_DELTA * (tf.abs(time_difference_error) - 0.5 * HUBER_LOSS_DELTA)
        loss = tf.where(cond, L2, L1)
        delta = tf.cast(time_difference_error < 0, tf.float32)
        return tf.reduce_mean(tf.abs(tau_hat - delta) * loss)
    return quantile_huber_loss


def np_huber_loss(y_true, y_pred):
    time_difference_error = y_true - y_pred

    cond = abs(time_difference_error) < HUBER_LOSS_DELTA
    L2 = 0.5 * time_difference_error**2
    L1 = HUBER_LOSS_DELTA * (abs(time_difference_error) - 0.5 * HUBER_LOSS_DELTA)

    loss = np.where(cond, L2, L1)

    return np.mean(loss)


def create_np_quantile_huber_loss(num_quantiles):
    tau = np.arange(num_quantiles + 1) / float(num_quantiles)

    def quantile_huber_loss(y_true, y_pred):
        tau_hat = 0.5 * (tau[1:] + tau[:-1])
        sorted_quantiles = np.argsort(y_pred)
        tau_hat = tau_hat[sorted_quantiles]
        time_difference_error = y_true - y_pred
        cond = abs(time_difference_error) < HUBER_LOSS_DELTA
        L2 = 0.5 * time_difference_error**2
        L1 = HUBER_LOSS_DELTA * (abs(time_difference_error) - 0.5 * HUBER_LOSS_DELTA)
        loss = np.where(cond, L2, L1)
        delta = [time_difference_error < 0]
        return np.mean(abs(tau_hat - delta) * loss)
    return quantile_huber_loss
