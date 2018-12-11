from typing import Tuple, Union

import keras


def encoder(
    input_observations: keras.layers.Input,
    input_shape: Union[Tuple[int], Tuple[int, int, int]],
    hidden_dim: int,
):
    if len(input_shape) == 1:
        encoded = keras.layers.Dense(
            units=hidden_dim, activation="relu", name="encoder/dense1"
        )(input_observations)
        encoded = keras.layers.Dense(
            units=hidden_dim, activation="relu", name="encoder/dense2"
        )(encoded)
    else:
        encoded = keras.layers.Conv2D(
            filters=16,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_first",
            activation="relu",
            name="encoder/conv1",
        )(input_observations)
        encoded = keras.layers.Conv2D(
            filters=32,
            kernel_size=4,
            strides=2,
            padding="same",
            data_format="channels_first",
            activation="relu",
            name="encoder/conv2",
        )(encoded)
        encoded = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            data_format="channels_first",
            activation="relu",
            name="encoder/conv3",
        )(encoded)
        encoded = keras.layers.Flatten()(encoded)
    return encoded
