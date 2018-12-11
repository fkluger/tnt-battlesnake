from typing import Tuple, Union

import keras

from common.tensorflow.noisy_dense import NoisyDense
from common.tensorflow.encoder import encoder


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

    encoded = encoder(input_observations, input_shape, hidden_dim)

    output = Dense(units=num_actions, activation="linear", name="output")(encoded)
    masked_output = keras.layers.Multiply()([output, input_actions])
    model = keras.Model(
        inputs=[input_observations, input_actions], outputs=masked_output
    )
    return model
