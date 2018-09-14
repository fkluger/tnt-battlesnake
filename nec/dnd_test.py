import tensorflow as tf
import numpy as np

from dnd import DifferentiableNeuralDictionary


class DifferentiableNeuralDictionaryTest(tf.test.TestCase):
    def testWrite(self):
        batch_size = 2
        key_length = 1
        capacity = 500000
        num_nearest_neighbours = 50
        dnd = DifferentiableNeuralDictionary(
            0, capacity, key_length, num_nearest_neighbours, 1e-3, 0.1
        )

        keys = tf.placeholder(
            name="keys", shape=[batch_size, key_length], dtype=tf.float32
        )
        values = tf.placeholder(name="values", shape=[batch_size], dtype=tf.float32)

        write_op = dnd.write(keys, values)
        lookup_op = dnd.lookup(keys)
        init_op = tf.global_variables_initializer()

        with self.test_session() as sess:

            sess.run(init_op)

            for i in range(10):
                sess.run(write_op, {keys: np.array([[i], [i]]), values: np.array([i, i])})

            sess.run(lookup_op, {keys: np.array([[1], [1]])})

            print("Keys\n", dnd.keys.eval())
            print("Values\n", dnd.values.eval())
            print("Ages\n", dnd.ages.eval())


if __name__ == "__main__":
    tf.test.main()
