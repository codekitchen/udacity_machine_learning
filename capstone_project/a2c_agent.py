"""Implements the A2C Agent"""

# pylint: disable=E1129

import tensorflow as tf
import numpy as np

from base_agent import BaseAgent
from layers import agent_network, fully_connected
from env_recorder import EnvRecorder


class A2CAgent(BaseAgent):
    """An Actor-Critic Advantage Network Agent"""

    def __init__(self, env_factory, state_dir):
        self.env_count = 6
        self.t_steps = 5
        self.gamma = 0.99
        self.total_steps = 40e6
        self.starting_lr = 1e-3
        self.envs = []
        for _ in range(self.env_count):
            env = env_factory()
            self.envs.append(env)
            env.info_cb = self.status
        super().__init__(self.envs[0], state_dir)
        print("A2C input shape ", self.state_shape)

    def _build_model(self):
        super()._build_model()
        with self.graph.as_default():
            with tf.variable_scope('network'):
                input_layer, final_layer, predict_layer = agent_network(
                    state_shape=self.state_shape,
                    image_input=self.image_input,
                    action_count=self.action_count)
                self.input_layer = input_layer
                self.predict_layer = predict_layer
                self.softmax_predict = tf.nn.softmax(self.predict_layer)
                self.value_layer = fully_connected(
                    final_layer, 1, activation=None, name="value")
            with tf.variable_scope('state'):
                self._frames = tf.Variable(0, trainable=False, name='frames', dtype=tf.int64)
                tf.summary.scalar('frames', self._frames)
                self.update_frames = tf.assign_add(self._frames, tf.cast(tf.shape(self.input_layer)[0], tf.int64))
                self.learning_rate = tf.maximum(0.0, self.starting_lr * (1.0 - (tf.cast(self._frames, tf.float32) / self.total_steps)))
                tf.summary.scalar('learning_rate', self.learning_rate)
            with tf.variable_scope('training'):
                self.target_predict = tf.placeholder(
                    tf.int32, shape=[None], name='target_predict')
                self.target_value = tf.placeholder(
                    tf.float32, shape=[None], name='target_value')
                # mse_predict = tf.reduce_mean(tf.squared_difference(self.predict_layer, self.target_predict))
                # tf.summary.scalar('mse_predict', mse_predict)
                mse_value = tf.reduce_mean(tf.squared_difference(
                    self.value_layer, self.target_value))
                tf.summary.scalar('mse_value', mse_value)
                diff_predict = tf.reduce_mean(
                    (self.target_value - self.value_layer) *
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.softmax_predict, labels=self.target_predict))
                tf.summary.scalar('err_predict', diff_predict)
                trainer = tf.train.RMSPropOptimizer(self.learning_rate)
                train_value = diff_predict + mse_value
                tf.summary.scalar('combined_err', train_value)
                self.train_op = trainer.minimize(
                    train_value, global_step=self._step)
                # self.train_op_value = trainer.minimize(
                #     mse_value, global_step=self._step)
                # self.train_op_predict = trainer.minimize(
                #     diff_predict, global_step=self._step)
            with tf.variable_scope('stats'):
                self.score_placeholder = tf.placeholder(
                    tf.float32, shape=[], name='score_input')
                score_1 = tf.Variable(0., trainable=False, name='score_1')
                tf.summary.scalar('score_1', score_1)
                score_10 = tf.Variable(0., trainable=False, name='score_10')
                tf.summary.scalar('score_10', score_10)
                score_100 = tf.Variable(0., trainable=False, name='score_100')
                tf.summary.scalar('score_100', score_100)

                self.set_scores = tf.group(
                    tf.assign(score_1, self.score_placeholder),
                    tf.assign(
                        score_10, score_10 + (self.score_placeholder / 10.0) - (score_10 / 10.0)),
                    tf.assign(
                        score_100, score_100 + (self.score_placeholder / 100.0) - (score_100 / 100.0)),
                )

    @property
    def frames(self):
        return self.session.run(self._frames)

    def status(self, env: EnvRecorder):
        score = env.ep_score
        _, frame, step, learning_rate = self.session.run(
            [self.set_scores, self._frames, self._step, self.learning_rate],
            {self.score_placeholder: score})
        return [
            ('frame', frame),
            ('step', step),
            ('lr', '{:.2e}'.format(learning_rate)),
        ]

    def act(self, states):
        """Returns action for each given state"""
        preds, _ = self.session.run(
            [self.softmax_predict, self.update_frames],
            {self.input_layer: states})
        # return np.argmax(preds, axis=1)
        return [np.random.choice(self.action_count, p=pred) for pred in preds]

    def value(self, states):
        """Returns predicted value for each given state"""
        vals = self.session.run(self.value_layer, {self.input_layer: states})
        return np.squeeze(vals, axis=1)

    def fit(self, states, actions, rewards):
        _, summary = self.session.run(
            [self.train_op, self.summary_data],
            {self.input_layer: states,
             self.target_value: rewards,
             self.target_predict: actions})
        self.writer.add_summary(summary, self.step)

    def run(self):
        states = [env.reset() for env in self.envs]
        last_save = 0
        while self.frames < self.total_steps:
            if self.frames - last_save > 100000:
                self._save()
                last_save = self.frames
            all_states = []
            all_actions = []
            all_rewards = []
            for envid, env in enumerate(self.envs):
                rewards = []
                actions = []
                seen = []
                for _ in range(self.t_steps):
                    seen.append(states[envid])
                    action = self.act(states[envid][None, :])[0]
                    actions.append(action)
                    next_state, reward, done, reset = env.step(action)
                    rewards.append(reward)
                    states[envid] = next_state
                    if reset:
                        states[envid] = env.reset()
                        break
                total_rewards = [0] * len(rewards)
                if not done:
                    total_rewards[-1] = self.value(next_state[None, :])[0]
                for idx in range(len(rewards) - 2, -1, -1):
                    total_rewards[idx] = rewards[idx] + \
                        self.gamma * total_rewards[idx + 1]
                assert len(seen) == len(actions)
                assert len(seen) == len(total_rewards)
                all_states += seen
                all_actions += actions
                all_rewards += total_rewards

            self.fit(all_states, all_actions, all_rewards)

        for env in self.envs:
            env.close()
        self._save()
