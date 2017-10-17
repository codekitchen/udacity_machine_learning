"""Implements the DQN Agent"""

#pylint: disable=E1129

import random
import time

import tensorflow as tf
import numpy as np

from layers import agent_network, NpMemory
from base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """A Deep Q-Network learning agent"""

    def __init__(self, env_factory, state_dir):
        self.env = env_factory()
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999954
        self.learning_rate = 0.00025
        self.replay_start_size = 50000
        self.batch_size = 32
        self.memory_size = 1000000
        self.target_update_frequency = 10000
        self.action_repeat = 4
        self.update_frequency = 4
        self.actions = 0
        self.frames = 0
        self.last_action = None
        self._gamma_value = None
        self.do_replay = False
        super().__init__(self.env, state_dir)
        print("DQN input shape ", self.state_shape)

    def _build_network(self):
        input_layer, _final_layer, predict_layer = agent_network(
            state_shape=self.state_shape,
            image_input=self.image_input,
            action_count=self.action_count)
        return input_layer, predict_layer

    def _build_model(self, epsilon=1.0, gamma=0.99):
        super()._build_model()
        with self.graph.as_default():
            with tf.variable_scope("network"):
                self.network = self._build_network()
            with tf.variable_scope("target_network"):
                self.target_network = self._build_network()
                with tf.name_scope("updater"):
                    # build the "updater" by iterating over all trainables in the target network,
                    # and creating an op to copy that same var from "network" to "target_network"
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
                state_dtype = np.uint8 if self.image_input else np.float32
                self.memory = NpMemory(self.memory_size, [
                    ('state', self.state_shape, state_dtype),
                    ('action', [], np.float32),
                    ('reward', [], np.float32),
                    ('done', [], np.bool)])
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
        super()._start_session()
        self.session.run(self._update_target_network)

    def _restore(self, checkpoint):
        super()._restore(checkpoint)
        self.memory = self.memory.restore(checkpoint)

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
        if step % 5000 == 0:
            self._save()
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

    def _save_other(self, save_path):
        self.memory.save(save_path)

    def act(self, state):
        """Tell this agent to choose an action and return the action chosen"""
        if self.memory.count(self.session) < self.replay_start_size:
            action = random.randrange(self.action_count)
        else:
            epsilon = self.epsilon
            if self.frames % self.action_repeat != 0:
                # action repeat
                action = self.last_action
            else:
                # select an action
                if random.random() <= epsilon:
                    # random exploration
                    action = random.randrange(self.action_count)
                else:
                    # selection based on network
                    action = np.argmax(self.predict(
                        state[None, :], self.network)[0])
                self.actions += 1
                if self.actions % self.update_frequency == 0:
                    self.do_replay = True
            # update epsilon each frame
            self.session.run(self._epsilon_assign, {self._epsilon_new_val: max(
                self.epsilon_min, epsilon * self.epsilon_decay)})
            self.frames += 1
        self.last_action = action
        return action

    def reward(self, state, action, reward, _next_state, done):
        """"Log the reward for the selected action after the envirnoment has been updated"""
        self._remember(state, action, reward, done)
        if self.do_replay:
            self._replay()
            self.do_replay = False

    def _remember(self, state, action, reward, done):
        self.memory.remember(
            self.session, (state, action, reward, float(done)))

    def _replay(self):
        rows = self.memory.sample(self.session, self.batch_size)
        if not rows:
            return
        states, actions, rewards, dones, next_states = rows
        pred_states = self.predict(states, self.network)
        pred_next_states = self.predict(next_states, self.target_network)
        t_rewards = rewards + (1.0 - dones) * self.gamma * \
            np.amax(pred_next_states, axis=1)
        pred_states[range(len(pred_states)), actions.astype(int)] = t_rewards
        self.fit(states, pred_states)

    def status(self, env):
        per_frame = (time.time() - env.start_time) / env.ep_frame
        return [
            ('frame', self.frames),
            ('action', self.actions),
            ('step', self.step),
            ('perframe', '{:.3f}'.format(per_frame)),
        ]

    def run(self):
        self.env.info_cb = self.status
        while True:
            state = self.env.reset()
            reset = False
            while not reset:
                action = self.act(state)
                next_state, reward, done, reset = self.env.step(action)
                self.reward(state, action, reward, next_state, done)
                state = next_state
