"""Defines NN layers"""
import random
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


class Column:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

    def build(self, size, index):
        self._storage = tf.get_variable(
            self.name, [size] + self.shape, initializer=tf.zeros_initializer, trainable=False)
        self.remember_value = tf.placeholder(
            tf.float32, shape=self.shape, name="{}_remember_value".format(self.name))
        self.remember = tf.scatter_update(
            self._storage, index, self.remember_value, name="{}_remember".format(self.name))
        self.gather_indices = tf.placeholder(
            tf.int64, name="{}_gather_indicies".format(self.name))
        self.gather = tf.gather(
            self._storage, self.gather_indices, name="{}_gather".format(self.name))
        return self


class Memory:
    def __init__(self, size, shapes):
        self.size = size
        self.shapes = shapes
        self._build()

    def _build(self):
        self._index = tf.get_variable(
            "index", [], dtype=tf.int64, initializer=tf.zeros_initializer, trainable=False)
        self._count = tf.get_variable(
            "count", [], dtype=tf.int64, initializer=tf.zeros_initializer, trainable=False)
        self.parts = [Column(name, shape).build(self.size, self._index)
                      for (name, shape) in self.shapes]
        with tf.control_dependencies([col.remember for col in self.parts]):
            self._bump_index = tf.assign(self._index, tf.mod(self._index + 1, self.size),
                                         name="bump_index")
            self._bump_count = tf.assign_add(self._count, 1, name="bump_count")

    def remember(self, session, values):
        ops = [col.remember for col in self.parts] + [
            self._bump_index,
            self._bump_count,
        ]
        session.run(ops, {col.remember_value: val for col,
                          val in zip(self.parts, values)})

    def sample(self, session, batch_size):
        count = int(session.run(self._count))
        rows = []
        indices = []
        if count >= self.size:
            count = self.size
        if count >= batch_size:
            indices = random.sample(range(count), batch_size)
            ops = [col.gather for col in self.parts]
            rows = session.run(
                ops,
                {col.gather_indices: indices for col in self.parts})
        return rows
