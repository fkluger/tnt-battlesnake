from typing import List

import tensorflow as tf
from tensorboard.plugins import projector
from dqn.huber_loss import huber_loss

from .dnd import DifferentiableNeuralDictionary


def build_graph(
    encoder: tf.keras.Model,
    optimizer: tf.train.Optimizer,
    dnds: List[DifferentiableNeuralDictionary],
    input_shape: List[int],
    num_actions: int,
):
    with tf.name_scope("nec"):
        observations_ph = tf.placeholder(
            name="observations", shape=[None] + input_shape, dtype=tf.int8
        )
        q_values_ph = tf.placeholder(name="q_values", shape=[None], dtype=tf.float32)
        actions_ph = tf.placeholder(name="actions", shape=[None], dtype=tf.int32)
        target_q_values_ph = tf.placeholder(
            name="target_q_values", shape=[None], dtype=tf.float32
        )

        keys = encoder(tf.to_float(observations_ph))
        q_values = tf.transpose([dnd.lookup(keys) for dnd in dnds], name="q_values")

        with tf.name_scope("compute_best_action"):
            mean_q_values = tf.reduce_mean(q_values)
            best_actions = tf.argmax(q_values, axis=1, name="best_actions")

        with tf.name_scope("loss"):
            q_values_selected = tf.reduce_sum(
                q_values * tf.one_hot(actions_ph, num_actions),
                axis=1,
                name="q_values_selected",
            )
            losses = tf.square(target_q_values_ph - q_values_selected)
            loss = tf.reduce_mean(losses)
            train_op = optimizer.minimize(loss)

        age_summaries = [
            tf.summary.histogram(f"dnd_{idx}/ages", dnd.ages)
            for idx, dnd in enumerate(dnds)
        ]

        embedding_variables = [dnd.keys for dnd in dnds]

        config = projector.ProjectorConfig()
        for idx, embedding_var in enumerate(embedding_variables):
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = f"metadata_{idx}.tsv"

        summaries = tf.summary.merge(age_summaries)

        def get_summaries():
            return tf.get_default_session().run(summaries)

        def get_values():
            return tf.get_default_session().run([dnd.values for dnd in dnds])

        def update_indices():
            return tf.get_default_session().run([dnd.update_index() for dnd in dnds])

        def train(observations, actions, target_q_values):
            error, q_values, errors, _ = tf.get_default_session().run(
                [loss, mean_q_values, losses, train_op],
                {
                    observations_ph: observations,
                    actions_ph: actions,
                    target_q_values_ph: target_q_values,
                },
            )
            return error, q_values, errors

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

        return train, act, write, update_indices, get_values, config, get_summaries
