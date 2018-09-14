import tensorflow as tf
import numpy as np

from dnd import DifferentiableNeuralDictionary


class DifferentiableNeuralDictionaryTest(tf.test.TestCase):
    def testWrite(self):
        batch_size = 2
        key_length = 1
        capacity = 20
        num_nearest_neighbours = 4
        dnd = DifferentiableNeuralDictionary(
            0, capacity, key_length, num_nearest_neighbours, 1e-3, 0.1
        )

        lookup_keys = tf.placeholder(
            name="keys", shape=[None, key_length], dtype=tf.float32
        )
        keys = tf.placeholder(
            name="keys", shape=[batch_size, key_length], dtype=tf.float32
        )
        values = tf.placeholder(name="values", shape=[batch_size], dtype=tf.float32)

        write_op = dnd.write(keys, values)
        lookup_op = dnd.lookup(lookup_keys)
        init_op = tf.global_variables_initializer()

        with self.test_session() as sess:

            sess.run(init_op)

            sess.run(dnd.update_index())

            for i in range(1, 5):
                print("Write")
                print(np.ones([batch_size, key_length]) * i)
                sess.run(write_op, {keys: np.ones([batch_size, key_length]) * i, values: np.ones([batch_size]) * i})
            
            sess.run(dnd.update_index())

            for i in range(100):
                print("Lookup")
                print(3 * np.ones([1, key_length]))
                value = sess.run(lookup_op, {lookup_keys: 3 * np.ones([1, key_length])})
                print(f"===> {value}")

            print("Keys\n", dnd.keys.eval())
            print("Values\n", dnd.values.eval())
            print("Ages\n", dnd.ages.eval())


if __name__ == "__main__":
    tf.test.main()
