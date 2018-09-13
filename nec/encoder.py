import tensorflow as tf


def create_encoder(input_shape, key_length):
    inputs = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.layers.Conv2D(32, 3, strides=3, activation="elu")(inputs)
    net = tf.keras.layers.Conv2D(64, 2, strides=2, activation="elu")(net)
    net = tf.keras.layers.Conv2D(64, 1, strides=1, activation="elu")(net)
    net = tf.keras.layers.Flatten()(net)
    outputs = tf.keras.layers.Dense(key_length, activation="elu")(net)
    return tf.keras.Model(inputs, outputs)

