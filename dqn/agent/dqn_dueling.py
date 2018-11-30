from typing import Tuple, Union

import keras
import tensorflow as tf

from .dqn import encode


def make_dqn_dueling(
    input_shape: Union[Tuple[int], Tuple[int, int, int]],
    hidden_dim: int,
    num_actions: int,
):
    input_observations = keras.layers.Input(
        shape=input_shape, name="input/observations"
    )
    input_actions = keras.layers.Input(shape=(num_actions,), name="input/actions")
    encoded = encode(input_observations, input_shape, hidden_dim)

    advantage = keras.layers.Dense(
        units=hidden_dim, activation="relu", name="advantage/dense1"
    )(encoded)
    advantage = keras.layers.Dense(
        units=num_actions, activation="linear", name="advantage/dense2"
    )(advantage)
    advantage = keras.layers.Lambda(
        lambda advt: advt - tf.reduce_mean(advt, axis=-1, keepdims=True),
        name="advantage/output",
    )(advantage)

    value = keras.layers.Dense(
        units=hidden_dim, activation="relu", name="value/dense1"
    )(encoded)
    value = keras.layers.Dense(units=1, activation="linear", name="value/dense2")(value)
    value = keras.layers.Lambda(
        lambda value: tf.tile(value, [1, num_actions]), name="value/output"
    )(value)

    output = keras.layers.Add()([value, advantage])
    masked_output = keras.layers.Multiply()([output, input_actions])
    model = keras.Model(
        inputs=[input_observations, input_actions], outputs=masked_output
    )
    return model

