import tensorflow as tf

HUBER_LOSS_DELTA = 2.0


def huber_loss(y_true, y_pred):
    with tf.name_scope("huber_loss"):
        err = y_true - y_pred

        cond = tf.abs(err) < HUBER_LOSS_DELTA
        L2 = 0.5 * tf.square(err)
        L1 = HUBER_LOSS_DELTA * (tf.abs(err) - 0.5 * HUBER_LOSS_DELTA)

        loss = tf.where(cond, L2, L1)

        return tf.reduce_mean(loss)
