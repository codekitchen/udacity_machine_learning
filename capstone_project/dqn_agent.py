"""Implements the DQN Agent"""
from collections import deque
import random

import tensorflow as tf
import numpy as np

from layers import fully_connected


class DQNAgent:
    """A Deep Q-Network learning agent"""

    def __init__(self, state_shape, action_count, batch_size, state_dir):
        self.action_count = action_count
        self.state_shape = [None] + list(state_shape)
        self.memory = deque(maxlen=20000)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99991  # 0.9999991
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.state_dir = state_dir
        self._gamma_value = None
        self._build_model(epsilon=1.0, gamma=0.99)
        self._start_session()

    def _build_model(self, epsilon, gamma):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._step = tf.train.create_global_step()
            self.input_layer = tf.placeholder(
                tf.float32, shape=self.state_shape, name="input")
            fc1 = fully_connected(self.input_layer, 512, name="dense1")
            self._predict = fully_connected(fc1, self.action_count,
                                            activation=None, name="output")
            with tf.variable_scope("training"):
                self.target = tf.placeholder(
                    tf.float32, shape=[None, self.action_count], name="target")
                mse = tf.reduce_mean(tf.squared_difference(
                    self._predict, self.target))
                tf.summary.scalar('mse', mse)
                self.train_op = tf.train.RMSPropOptimizer(
                    self.learning_rate).minimize(mse, global_step=self._step)
            self._epsilon = tf.Variable(
                epsilon, trainable=False, name="state/epsilon")
            self._gamma = tf.Variable(
                gamma, trainable=False, name="state/gamma")

    def _start_session(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.summary_data = tf.summary.merge_all()
            summary_dir = "{}/summary".format(self.state_dir)
            self.writer = tf.summary.FileWriter(summary_dir, self.graph)
            self.session.run(tf.global_variables_initializer())
            checkpoint = tf.train.latest_checkpoint(self._snapshot_dir)
            if checkpoint:
                self.saver.restore(self.session, checkpoint)

    def __del__(self):
        if self.session:
            self.session.close()

    def predict(self, state: np.ndarray):
        """Give propabilities for each action, given `state` and the current model"""

        prediction = self.session.run(
            self._predict, {self.input_layer: state[None, :]})
        return prediction[0]

    def fit(self, states, targets):
        args = {}
        step = self.step
        if step % 1000 == 0:
            snapshot_name = "{}/snap".format(self._snapshot_dir)
            self.saver.save(self.session, snapshot_name, global_step=step)
        if step % 100 == 0:
            args['options'] = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            args['run_metadata'] = tf.RunMetadata()
        _, summary = self.session.run([self.train_op, self.summary_data],
                                      {self.input_layer: states,
                                          self.target: targets},
                                      **args)
        if 'run_metadata' in args:
            self.writer.add_run_metadata(
                args['run_metadata'], 'step_{}'.format(step))
        self.writer.add_summary(summary, step)

    @property
    def epsilon(self):
        return self.session.run(self._epsilon)

    @property
    def gamma(self):
        self._gamma_value = self._gamma_value or self.session.run(self._gamma)
        return self._gamma_value

    @property
    def step(self):
        return tf.train.global_step(
            self.session, tf.train.get_global_step(graph=self.session.graph))

    @property
    def _snapshot_dir(self):
        return "{}/snapshots".format(self.state_dir)

    def act(self, state):
        """Tell this agent to choose an action and return the action chosen"""
        epsilon = self.epsilon
        if random.random() <= epsilon:
            action = random.randrange(self.action_count)
        else:
            action = np.argmax(self.predict(state))
        self._epsilon.assign(
            max(self.epsilon_min, epsilon * self.epsilon_decay))
        return action

    def reward(self, state, action, reward, next_state, done):
        """"Log the reward for the selected action after the envirnoment has been updated"""
        self._remember(state, action, reward, next_state, done)
        self._replay()

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.predict(next_state)))
            target_f = self.predict(state)
            target_f[action] = target
            states.append(state)
            targets.append(target_f)
        self.fit(states, targets)
