"""Implements the DQN Agent"""
import random

import tensorflow as tf
import numpy as np

from layers import fully_connected, Memory


class DQNAgent:
    """A Deep Q-Network learning agent"""

    def __init__(self, state_shape, action_count, batch_size, state_dir):
        self.action_count = action_count
        self.state_shape = list(state_shape)
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
                tf.float32, shape=([None] + self.state_shape), name="input")
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
            with tf.variable_scope("memory"):
                self.memory = Memory(20000, [
                    ('state', self.state_shape),
                    ('action', []),
                    ('reward', []),
                    ('next_state', self.state_shape),
                    ('done', [])])
            with tf.variable_scope("state"):
                self._epsilon = tf.Variable(
                    epsilon, trainable=False, name="epsilon")
                self._epsilon_new_val = tf.placeholder(
                    tf.float32, shape=(), name="epsilon_update_val")
                self._epsilon_assign = tf.assign(
                    self._epsilon, self._epsilon_new_val, name="epsilon_update")
                self._gamma = tf.Variable(
                    gamma, trainable=False, name="gamma")

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
            self._predict, {self.input_layer: state})
        return prediction

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
            action = np.argmax(self.predict(state[None, :])[0])
        # self.session.run(self._epsilon.assign(
            # max(self.epsilon_min, epsilon * self.epsilon_decay)))
        self.session.run(self._epsilon_assign, {self._epsilon_new_val: max(
            self.epsilon_min, epsilon * self.epsilon_decay)})
        return action

    def reward(self, state, action, reward, next_state, done):
        """"Log the reward for the selected action after the envirnoment has been updated"""
        self._remember(state, action, reward, next_state, done)
        self._replay()

    def _remember(self, state, action, reward, next_state, done):
        self.memory.remember(self.session, (state, action,
                                            reward, next_state, float(done)))

    def _replay(self):
        rows = self.memory.sample(self.session, self.batch_size)
        if (len(rows) < 1):
            return
        states = []
        targets = []
        rows.append(self.predict(rows[0]))
        rows.append(self.predict(rows[3]))
        rows = [row for row in zip(*rows)]
        for state, action, reward, _next_state, done, state_p, next_state_p in rows:
            action = int(action)
            done = bool(done)
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(next_state_p))
            target_f = state_p
            target_f[action] = target
            states.append(state)
            targets.append(target_f)
        if len(states) > 0:
            self.fit(states, targets)
