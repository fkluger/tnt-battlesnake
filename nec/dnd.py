import tensorflow as tf
import faiss


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
        self._reset_index()
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

    def update_index(self):
        def _update_index(keys):
            self.index.reset()
            self.index.add(keys)
            return True
        return tf.py_func(_update_index, [self.keys], bool)

    def lookup(self, keys):
        with tf.name_scope("lookup"):
            # [batch_size, num_nearest_neighbours]
            nn_distances, nn_indices = self._find_nearest_neighbours(keys)

            update_ages_op = self._update_ages(nn_indices)

            nn_values = tf.nn.embedding_lookup(self.values, nn_indices)

            nn_distances = 1. / (nn_distances + self.delta)

            weights = nn_distances / tf.reshape(
                tf.reduce_sum(nn_distances, -1), [-1, 1]
            )
            values = tf.reduce_sum(nn_values * weights, axis=-1)

            with tf.control_dependencies([update_ages_op, self.write(keys, values)]):
                return tf.identity(values)

    def write(self, keys, values):
        with tf.name_scope("write"):
            nn_distances, nn_indices = self._find_nearest_neighbours(keys)
            # [batch_size] of minimal distances to self.values
            min_distances = nn_distances[:, 0]

            zero_distances_mask = tf.equal(min_distances, 0.)
            not_zero_distances_mask = tf.logical_not(tf.equal(min_distances, 0.))

            update_values_op = self._update_values(
                tf.boolean_mask(nn_indices, zero_distances_mask),
                tf.boolean_mask(nn_distances, zero_distances_mask),
            )

            # Update values when min_distances is evaluated
            with tf.control_dependencies([update_values_op]):
                keys = tf.identity(keys)

            new_keys, new_values = (
                tf.boolean_mask(keys, not_zero_distances_mask),
                tf.boolean_mask(values, not_zero_distances_mask),
            )

            return  tf.cond(
                tf.less_equal(
                    tf.add(self.pointer, tf.shape(new_keys)[0]), self.capacity
                ),
                lambda: self._append(new_keys, new_values),
                lambda: self._replace(new_keys, new_values),
            )

    def _replace(self, keys, values):
        with tf.name_scope("replace"):
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
        with tf.name_scope("append"):
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
        with tf.name_scope("update_values"):
            values_to_update = tf.gather(self.values, indices)
            updated_values = values_to_update + self.learning_rate * (
                values - values_to_update
            )
            return tf.scatter_update(self.values, indices, updated_values)

    def _reset_index(self):
        self.index = faiss.IndexFlatL2(self.key_length)

    def _find_nearest_neighbours(self, keys):
        with tf.name_scope("find_nearest_neighbours"):
            def approximate_nearest_neighbours(query_keys):
                return self.index.search(query_keys, self.num_nearest_neighbours)
            nn_distances, nn_indices = tf.py_func(approximate_nearest_neighbours, [keys], [tf.float32, tf.int64])
            nn_distances.set_shape([None, self.num_nearest_neighbours])
            nn_indices.set_shape([None, self.num_nearest_neighbours])
            return nn_distances, nn_indices

    def _update_ages(self, indices):
        with tf.name_scope("update_ages"):
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
