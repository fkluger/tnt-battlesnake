import tensorflow as tf
from tensorflow.python.framework import ops


@ops.RegisterGradient("NearestNeighbors")
def _NearestNeighborsGrad(op, grad, _):
    gradient = op.inputs[0] * tf.reduce_mean(grad, -1, keepdims=True)
    print(gradient)
    return gradient
