"""Defines NN layers"""
import tensorflow as tf


def fully_connected(input_tensor: tf.Tensor,
                    neuron_count: int = 10,
                    activation=tf.nn.relu,
                    name='dense') -> tf.Tensor:
    """
    Returns a fully connected NN layer

    Args:
        input_tensor: The input data for this layer.
        neuron_count: The number of output values (neurons in the layer).
    Returns:
        The new fully connected layer.
    """

    with tf.variable_scope(None, name):
        input_shape = input_tensor.shape
        input_shape.assert_has_rank(2)
        weights = tf.get_variable("weights",
                                  [input_shape[1].value, neuron_count],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",
                                 [neuron_count],
                                 initializer=tf.constant_initializer(0.1))
        layer = input_tensor @ weights + biases
        if activation:
            layer = activation(layer)
    return layer
