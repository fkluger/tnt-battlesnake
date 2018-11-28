import tensorflow as tf


def huber_loss(y_true, y_pred):
    with tf.name_scope("huber_loss"):
        err = y_true - y_pred

        cond = tf.abs(err) < 1.0
        L2 = 0.5 * tf.square(err)
        L1 = tf.abs(err) - 0.5

        loss = tf.where(cond, L2, L1)

        return tf.reduce_mean(loss)
