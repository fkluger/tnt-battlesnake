import tensorflow as tf


class DifferentiableNeuralDictionary:
    def __init__(
        self,
        action_index: int,
        capacity: int,
        key_length: int,
        num_nearest_neighbours: int,
        delta: float,
        learning_rate: float,
    ):
        self.action_index = action_index
        self.capacity = capacity
        self.key_length = key_length
        self.num_nearest_neighbours = num_nearest_neighbours
        self.delta = delta
        self.learning_rate = learning_rate
        with tf.name_scope("dnd"):
            self.pointer = tf.get_variable(
                f"pointer_{action_index}",
                [],
                dtype=tf.int32,
                initializer=tf.initializers.zeros(dtype=tf.int32),
                trainable=False,
            )
            self.keys = tf.get_variable(
                f"keys_{action_index}",
                [self.capacity, self.key_length],
                dtype=tf.float32,
                initializer=tf.initializers.zeros(),
                trainable=True,
            )
            self.values = tf.get_variable(
                f"values_{action_index}",
                [self.capacity],
                dtype=tf.float32,
                initializer=tf.initializers.zeros(),
                trainable=True,
            )
            self.ages = tf.get_variable(
                f"ages_{action_index}",
                [self.capacity],
                dtype=tf.int32,
                initializer=tf.initializers.zeros(dtype=tf.int32),
                trainable=False,
            )

    def update(self, keys, values, ages):
        with tf.name_scope("update"):
            return tf.group(
                tf.assign(self.keys, keys),
                tf.assign(self.values, values),
                tf.assign(self.ages, ages),
            )

    def lookup(self, keys):
        with tf.name_scope("lookup"):
            # [batch_size, capacity]
            distances = self._distance_kernel(keys)
            # [batch_size, num_nearest_neighbours]
            nn_distances, nn_indices = tf.nn.top_k(
                distances, k=self.num_nearest_neighbours
            )

            update_ages_op = self._update_ages(nn_indices)

            # Update ages when nn_values is evaluated
            with tf.control_dependencies([update_ages_op]):
                nn_values = tf.nn.embedding_lookup(self.values, nn_indices)

            weights = nn_distances / tf.reshape(
                tf.reduce_sum(nn_distances, -1), [-1, 1]
            )
            values = tf.reduce_sum(nn_values * weights, axis=-1)

            with tf.control_dependencies([self.write(keys, values)]):
                return values

    def write(self, keys, values):
        with tf.name_scope("write"):
            # batch_size = tf.shape(keys)[0]
            # # [batch_size, capacity]
            # distances = self._distance_kernel(keys)
            # # [batch_size] of indices of self.keys
            # min_distance_indices = tf.argmax(distances, axis=-1, output_type=tf.int32)
            # # [batch_size] of minimal distances to self.values
            # min_distances = tf.gather_nd(
            #     distances,
            #     tf.stack([tf.range(batch_size), min_distance_indices], axis=1),
            # )

            # zero_distances_mask = tf.equal(min_distances, 1. / self.delta)
            # not_zero_distances_mask = tf.logical_not(tf.equal(min_distances, 0.))

            # update_values_op = self._update_values(
            #     tf.boolean_mask(min_distance_indices, zero_distances_mask),
            #     tf.boolean_mask(min_distances, zero_distances_mask),
            # )

            # new_keys, new_values = (
            #     tf.boolean_mask(keys, not_zero_distances_mask),
            #     tf.boolean_mask(values, not_zero_distances_mask),
            # )

            new_keys, new_values = keys, values

            condition_pointer_less_capacity = tf.cond(
                tf.less_equal(
                    tf.add(self.pointer, tf.shape(new_keys)[0]), self.capacity
                ),
                lambda: self._append(new_keys, new_values),
                lambda: self._replace(new_keys, new_values),
            )

            # # Update values when min_distances is evaluated
            # with tf.control_dependencies([update_values_op]):
            #     condition_pointer_less_capacity = tf.identity(
            #         condition_pointer_less_capacity
            #     )

            return condition_pointer_less_capacity

    def _replace(self, keys, values):
        _, max_age_indices = tf.nn.top_k(self.ages, k=tf.shape(keys)[0], sorted=False)
        replace_op = tf.group(
            tf.scatter_update(self.keys, max_age_indices, keys),
            tf.scatter_update(self.values, max_age_indices, values),
        )
        with tf.control_dependencies([replace_op]):
            return tf.scatter_update(
                self.ages,
                max_age_indices,
                tf.zeros(tf.shape(max_age_indices), dtype=tf.int32),
            )

    def _append(self, keys, values):
        append_op = tf.group(
            tf.scatter_update(
                self.keys,
                tf.range(self.pointer, self.pointer + tf.shape(keys)[0]),
                keys,
            ),
            tf.scatter_update(
                self.values,
                tf.range(self.pointer, self.pointer + tf.shape(values)[0]),
                values,
            ),
        )
        with tf.control_dependencies([append_op]):
            return tf.assign_add(self.pointer, tf.shape(keys)[0])

    def _update_values(self, indices, values):
        values_to_update = tf.gather(self.values, indices)
        updated_values = values_to_update + self.learning_rate * (
            values - values_to_update
        )
        return tf.scatter_update(self.values, indices, updated_values)

    def _distance_kernel(self, keys):
        distances = tf.square(
            tf.norm(
                tf.expand_dims(self.keys, 0)  # [1, capacity, key_length]
                - tf.expand_dims(keys, 1),  # [batch_size, 1, key_length]
                axis=-1,
            )
        )
        return 1. / (distances + self.delta)

    def _update_ages(self, indices):
        indices_flatten, _ = tf.unique(tf.reshape(indices, [-1]))
        increment_ages = tf.assign_add(
            self.ages, tf.ones(tf.shape(self.ages), dtype=tf.int32)
        )
        with tf.control_dependencies([increment_ages]):
            reset_indices = tf.scatter_update(
                self.ages,
                indices_flatten,
                tf.zeros(tf.shape(indices_flatten), dtype=tf.int32),
            )
        return reset_indices
