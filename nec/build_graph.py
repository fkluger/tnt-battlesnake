from typing import List, Tuple

import tensorflow as tf
from .dnd import DifferentiableNeuralDictionary


def build_graph(
    encoder: tf.keras.Model,
    optimizer: tf.train.Optimizer,
    dnds: List[DifferentiableNeuralDictionary],
    input_shape: List[int],
    num_actions: int
):
    with tf.variable_scope("nec", reuse=None):
        observations_ph = tf.placeholder(
            name="observations", shape=[None] + input_shape, dtype=tf.int8
        )
        q_values_ph = tf.placeholder(name="q_values", shape=[None], dtype=tf.float32)
        actions_ph = tf.placeholder(name="actions", shape=[None], dtype=tf.int32)
        target_q_values_ph = tf.placeholder(
            name="target_q_values", shape=[None], dtype=tf.float32
        )

        keys = encoder(tf.to_float(observations_ph))

        q_values = [dnd.lookup(keys) for dnd in dnds]
        best_actions = tf.argmax(tf.transpose(q_values), axis=1, name="best_actions")

        q_values_selected = tf.gather(q_values, actions_ph, name="q_values_selected")
        loss = tf.reduce_sum(tf.square(target_q_values_ph - q_values_selected))
        train_op = optimizer.minimize(loss)

        def update_indices():
            return tf.get_default_session().run(
                [dnd.update_index() for dnd in dnds]
            )

        def train(observations, actions, target_q_values):
            error, _ = tf.get_default_session().run(
                [loss, train_op],
                {
                    observations_ph: observations,
                    actions_ph: actions,
                    target_q_values_ph: target_q_values,
                },
            )
            return error

        def act(observations):
            return tf.get_default_session().run(
                [best_actions, q_values], {observations_ph: observations}
            )

        def write(observations_per_action, q_values_per_action):
            return [
                tf.get_default_session().run(
                    dnds[a].write(keys, q_values_ph),
                    {
                        observations_ph: observations_per_action[a],
                        q_values_ph: q_values_per_action[a],
                    },
                )
                for a in range(num_actions)
                if len(observations_per_action[a])
            ]

        return train, act, write, update_indices
