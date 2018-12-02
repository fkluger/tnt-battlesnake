from typing import Tuple, Union

import keras

from common.tensorflow.noisy_dense import NoisyDense


def encode(
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
            filters=32,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_first",
            input_shape=input_shape,
            activation="relu",
            name="encoder/conv1",
        )(input_observations)
        encoded = keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            padding="same",
            data_format="channels_first",
            activation="relu",
            name="encoder/conv2",
        )(encoded)
        encoded = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            data_format="channels_first",
            activation="relu",
            name="encoder/conv3",
        )(encoded)
        encoded = keras.layers.Flatten()(encoded)
    return encoded


def make_dqn(
    input_shape: Union[Tuple[int], Tuple[int, int, int]],
    hidden_dim: int,
    num_actions: int,
    use_noisy_dense_layers=False,
):
    input_observations = keras.layers.Input(
        shape=input_shape, name="input/observations"
    )
    input_actions = keras.layers.Input(shape=(num_actions,), name="input/actions")

    Dense = NoisyDense if use_noisy_dense_layers else keras.layers.Dense

    encoded = encode(input_observations, input_shape, hidden_dim)

    output = Dense(units=num_actions, activation="linear", name="output")(encoded)
    masked_output = keras.layers.Multiply()([output, input_actions])
    model = keras.Model(
        inputs=[input_observations, input_actions], outputs=masked_output
    )
    return model
