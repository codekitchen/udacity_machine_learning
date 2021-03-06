"""Defines NN layers"""
from pickle import Pickler, Unpickler
from pathlib import Path
from macos_file import MacOSFile

import random
import tensorflow as tf
import numpy as np


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


def conv2d(input_tensor: tf.Tensor,
           n_filters: int,
           filter_size: int,
           stride: int,
           activation=tf.nn.relu,
           name='conv2d') -> tf.Tensor:
    """
    Returns a 2D convolution layer

    Args:
        input_tensor: The input data for this layer. Must be 4D, of the shape [None, W, H, D] 
        n_filters: The number of filters (feature channels)) to generate.
        filter_size: The filter size as an int (will be square).
        stride: The stride of the filter, affects output dimensions.
    Returns:
        The new conv2d layer.
    """

    with tf.variable_scope(None, name):
        # input_shape = input_tensor.shape
        # input_shape.assert_has_rank(4)
        # in_channels = input_shape[3]
        # layer = tf.nn.conv2d(input_tensor,
        #                      [filter_size, filter_size, in_channels.value, n_filters], [1, stride, stride, 1], "SAME", name="filter")
        # if activation:
        #     layer = activation(layer)
        layer = tf.layers.conv2d(inputs=input_tensor, filters=n_filters,
                                 kernel_size=filter_size, strides=stride, activation=activation, name=name)
        return layer

def agent_network(state_shape, action_count, image_input):
    input_layer = tf.placeholder(
        tf.float32, shape=([None] + state_shape), name="input")
    if image_input:
        # conv network to process images
        conv = conv2d(tf.cast(input_layer, tf.float32)/255., 32, 8, 4, name="conv1")
        conv = conv2d(conv, 64, 4, 2, name="conv2")
        if conv.shape[1] > 3:
            conv = conv2d(conv, 64, 3, 1, name="conv3")
        flat_shape = np.prod([dim.value for dim in conv.shape[1:]])
        layer = tf.reshape(conv, [-1, flat_shape])
    else:
        layer = input_layer
    layer = fully_connected(layer, 512, name="dense1")
    # layer = fully_connected(layer, 512, name="dense2")
    predict = fully_connected(layer, action_count,
                                activation=None, name="output")
    return input_layer, layer, predict

class Column:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

    def build(self, size, index, dtype):
        with tf.variable_scope(self.name):
            self._storage = tf.get_variable(
                "storage", [size] + self.shape, initializer=tf.zeros_initializer, trainable=False, dtype=dtype)
            self.remember_value = tf.placeholder(
                dtype, shape=self.shape, name="remember_value")
            self.remember = tf.scatter_update(
                self._storage, index, self.remember_value, name="remember")
            self.gather_indices = tf.placeholder(
                tf.int64, name="gather_indicies", shape=[None])
            self.gather = tf.gather(
                self._storage, self.gather_indices, name="gather")
            return self


class Memory:
    def __init__(self, size, shapes):
        self.size = size
        self.shapes = shapes
        self._build()

    def _build(self):
        self._index = tf.get_variable(
            "index/index", [], dtype=tf.int64, initializer=tf.zeros_initializer, trainable=False)
        self._count = tf.get_variable(
            "count/count", [], dtype=tf.int64, initializer=tf.zeros_initializer, trainable=False)
        self.parts = [Column(name, shape).build(self.size, self._index, dtype)
                      for (name, shape, dtype) in self.shapes]
        with tf.variable_scope("index"):
            inc = tf.mod(self._index + 1, self.size)
            with tf.control_dependencies([col.remember for col in self.parts]):
                self._bump_index = tf.assign(
                    self._index, inc, name="bump_index")
        with tf.variable_scope("count"):
            self._bump_count = tf.assign_add(
                self._count, 1, name="bump_count")

    def restore(self, path):
        return self

    def save(self, path):
        pass

    def count(self, session):
        return int(session.run(self._count))

    def remember(self, session, values):
        ops = [col.remember for col in self.parts] + [
            self._bump_index,
            self._bump_count,
        ]
        session.run(ops, {col.remember_value: val for col,
                          val in zip(self.parts, values)})

    def sample(self, session, batch_size):
        count = self.count(session)
        indices = []
        if count < batch_size:
            return
        if count >= self.size:
            count = self.size
        indices = random.sample(range(count), batch_size)
        ops = [col.gather for col in self.parts]
        return session.run(
            ops,
            {col.gather_indices: indices for col in self.parts})


class NpColumn:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

    def build(self, size, dtype):
        self.storage = np.zeros([size] + self.shape, dtype=dtype)
        return self


class NpMemory:
    """assumes that the first shape is the input state"""
    def __init__(self, size, shapes):
        self.size = size
        self.shapes = shapes
        self._index = 0
        self._count = 0
        self._last_insert = 0
        self.storage = [NpColumn(name, shape).build(self.size, dtype)
                        for (name, shape, dtype) in self.shapes]

    @staticmethod
    def fname(path):
        return path + ".memory"

    @classmethod
    def restore(self, path):
        with open(self.fname(path), 'rb') as f:
            p = Unpickler(f)
            return p.load()

    def save(self, path):
        with open(self.fname(path), 'wb') as f:
            p = Pickler(MacOSFile(f), -1)
            p.dump(self)
        # remove old *.memory files
        save_dir = Path(path).parent
        for memfile in save_dir.glob("*.memory"):
            basefile = memfile.with_suffix('.meta')
            if not basefile.is_file():
                memfile.unlink()

    def count(self, _session):
        return self._count

    def remember(self, _session, values):
        for col, val in zip(self.storage, values):
            col.storage[self._index] = val
        self._count += 1
        self._last_insert = self._index
        self._index = (self._index + 1) % self.size

    def sample(self, _session, batch_size):
        """returns the next state for each sample as well"""
        count = self._count
        if count <= batch_size:
            return
        if count > self.size:
            count = self.size
        # we can't use random.sample(range(count)) here because
        # there's an element in the middle of the range
        # (self._last_insert) that we need to reject.
        indices = []
        while len(indices) < batch_size:
            n = random.randrange(count)
            if n != self._last_insert:
                indices.append(n)
        indices = np.array(indices)
        # indices = np.array(random.sample(range(count), batch_size))
        samples = [col.storage[indices] for col in self.storage]
        # add the next states as well
        next_indices = (indices + 1) % self.size
        samples.append(self.storage[0].storage[next_indices])
        return samples
