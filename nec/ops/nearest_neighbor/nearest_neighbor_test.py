import tensorflow as tf

nn_module = tf.load_op_library("../build/libnearest_neighbor.dylib")


class NearestNeighbourTest(tf.test.TestCase):
    def testWrite(self):

        with self.test_session() as sess:

            sess.run(tf.global_variables_initializer())
            nn_module.add_to_index([[1, 2], [3, 4]], reset=False, index_name="0").eval()
            distances, indices = nn_module.nearest_neighbors(
                [[3, 3], [1, 2]], k=1, index_name="0"
            )
            print(distances.eval())
            print(indices.eval())
            nn_module.add_to_index([[3, 3], [1, 2]], reset=True, index_name="0").eval()
            distances, indices = nn_module.nearest_neighbors(
                [[1, 2]], k=1, index_name="0"
            )
            print(distances.eval())
            print(indices.eval())


if __name__ == "__main__":
    tf.test.main()
