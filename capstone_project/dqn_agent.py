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
        self.memory_size = 1000000
        self.target_update_frequency = 2000
        self._gamma_value = None
        self._build_model(epsilon=1.0, gamma=0.99)
        self._start_session()

    def _build_network(self):
        input_layer = tf.placeholder(
            tf.float32, shape=([None] + self.state_shape), name="input")
        fc1 = fully_connected(input_layer, 512, name="dense1")
        fc2 = fully_connected(fc1, 512, name="dense2")
        predict = fully_connected(fc2, self.action_count,
                                  activation=None, name="output")
        return input_layer, predict

    def _build_model(self, epsilon, gamma):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._step = tf.train.create_global_step()
            with tf.variable_scope("network"):
                self.network = self._build_network()
            with tf.variable_scope("target_network"):
                self.target_network = self._build_network()
                with tf.name_scope("updater"):
                    tn_vars = self.graph.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network/")
                    updates = [tf.assign(var, self.graph.get_tensor_by_name(
                        var.name.replace("target_network/", "network/")), name="update_{}".format(var.name.replace("target_network/", "").replace("/", "_").split(":")[0])) for var in tn_vars]
                    self._update_target_network = tf.group(*updates)
            with tf.variable_scope("training"):
                self.target = tf.placeholder(
                    tf.float32, shape=[None, self.action_count], name="target")
                mse = tf.reduce_mean(tf.squared_difference(
                    self.network[1], self.target))
                tf.summary.scalar('mse', mse)
                self.train_op = tf.train.RMSPropOptimizer(
                    self.learning_rate).minimize(mse, global_step=self._step)
            with tf.variable_scope("memory"):
                self.memory = Memory(self.memory_size, [
                    ('state', self.state_shape),
                    ('action', []),
                    ('reward', []),
                    ('next_state', self.state_shape),
                    ('done', [])])
            with tf.variable_scope("state"):
                self._epsilon = tf.Variable(
                    epsilon, trainable=False, name="epsilon")
                tf.summary.scalar('epsilon', self._epsilon)
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
            self.session.run(self._update_target_network)
            self.graph.finalize()
            checkpoint = tf.train.latest_checkpoint(self._snapshot_dir)
            if checkpoint:
                self.saver.restore(self.session, checkpoint)

    def __del__(self):
        if self.session:
            self.session.close()

    def predict(self, state: np.ndarray, network):
        """Give propabilities for each action, given `state` and the current model"""

        prediction = self.session.run(network[1], {network[0]: state})
        return prediction

    def fit(self, states, targets):
        args = {}
        step = self.step
        if step % self.target_update_frequency == 0:
            self.session.run(self._update_target_network)
            print("copied network into target_network")
        if step % 1000 == 0:
            snapshot_name = "{}/snap".format(self._snapshot_dir)
            self.saver.save(self.session, snapshot_name, global_step=step)
        if step % 100 == 0:
            args['options'] = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            args['run_metadata'] = tf.RunMetadata()
        _, summary = self.session.run([self.train_op, self.summary_data],
                                      {self.network[0]: states,
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
            action = np.argmax(self.predict(state[None, :], self.network)[0])
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
        if not rows:
            return
        states, actions, rewards, next_states, dones = rows
        pred_states = self.predict(states, self.network)
        pred_next_states = self.predict(next_states, self.target_network)
        t_rewards = rewards + (1.0 - dones) * self.gamma * np.amax(pred_next_states, axis=1)
        pred_states[range(len(pred_states)), actions.astype(int)] = t_rewards
        self.fit(states, pred_states)