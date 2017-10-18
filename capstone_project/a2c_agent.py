"""Implements the A2C Agent"""

# pylint: disable=E1129

import time

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
        self.start_time = time.time()
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
                self.update_frames = tf.assign_add(self._frames,
                                                   tf.cast(tf.shape(self.input_layer)[0], tf.int64))
                lr_calc = self.starting_lr * \
                    (1.0 - (tf.cast(self._frames, tf.float64) / self.total_steps))
                self.learning_rate = tf.maximum(tf.cast(0.0, tf.float64), lr_calc)
                tf.summary.scalar('learning_rate', self.learning_rate)
            with tf.variable_scope('training'):
                self.target_predict = tf.placeholder(
                    tf.int32, shape=[None], name='target_predict')
                self.target_value = tf.placeholder(
                    tf.float32, shape=[None], name='target_value')
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
            with tf.variable_scope('stats'):
                self.score_placeholder = tf.placeholder(
                    tf.float32, shape=[], name='score_input')
                score_1 = tf.Variable(0., trainable=False, name='score_1')
                tf.summary.scalar('score_1', score_1)
                score_100 = tf.Variable(0., trainable=False, name='score_100')
                tf.summary.scalar('score_100', score_100)
                score_1000 = tf.Variable(0., trainable=False, name='score_1000')
                tf.summary.scalar('score_1000', score_1000)

                self.set_scores = tf.group(
                    tf.assign(score_1, self.score_placeholder),
                    tf.assign(
                        score_100,
                        score_100 + (self.score_placeholder / 100.0) - (score_100 / 100.0)),
                    tf.assign(
                        score_1000,
                        score_1000 + (self.score_placeholder / 1000.0) - (score_1000 / 1000.0)),
                )

    @property
    def frames(self):
        return self.session.run(self._frames)

    def status(self, env: EnvRecorder):
        score = env.ep_score
        _, frame, step, learning_rate = self.session.run(
            [self.set_scores, self._frames, self._step, self.learning_rate],
            {self.score_placeholder: score})
        per_frame = (time.time() - self.start_time) / frame
        return [
            ('frame', frame),
            ('step', step),
            ('perframe', '{:.6f}'.format(per_frame)),
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
        _, summary, step = self.session.run(
            [self.train_op, self.summary_data, self._step],
            {self.input_layer: states,
             self.target_value: rewards,
             self.target_predict: actions})

        if step % 50 == 0:
            self.writer.add_summary(summary, step)
    
    class RunEnv:
        def __init__(self, env):
            self.env = env
            self.done = False
            self.reset = True
            self.current_state = None
            self.states = []
            self.actions = []
            self.rewards = []

        @property
        def active(self):
            return not self.reset

        def prepare(self):
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            if self.reset:
                self.current_state = self.env.reset()
                self.reset = False

        def step(self, action):
            self.states.append(self.current_state)
            self.actions.append(action)
            next_state, reward, self.done, self.reset = self.env.step(action)
            self.rewards.append(reward)
            self.current_state = next_state

    def run(self):
        self.start_time = time.time()
        run_envs: [self.RunEnv] = [self.RunEnv(env) for env in self.envs]
        last_save = 0
        while self.frames < self.total_steps:
            if self.frames - last_save > 100000:
                self._save()
                last_save = self.frames

            all_states = []
            all_actions = []
            all_rewards = []

            for env in run_envs:
                env.prepare()

            for _ in range(self.t_steps):
                envs = [env for env in run_envs if env.active]
                states = [env.current_state for env in envs]
                actions = self.act(states)
                for env, action in zip(envs, actions):
                    env.step(action)
            values = self.value([env.current_state for env in run_envs])
            for env, value in zip(run_envs, values):
                total_rewards = [0] * len(env.rewards)
                if not env.done:
                    total_rewards[-1] = value
                for idx in range(len(env.rewards) - 2, -1, -1):
                    total_rewards[idx] = env.rewards[idx] + \
                        self.gamma * total_rewards[idx + 1]
                assert len(env.states) == len(env.actions)
                assert len(env.states) == len(total_rewards)
                all_states += env.states
                all_actions += env.actions
                all_rewards += total_rewards

            self.fit(all_states, all_actions, all_rewards)

        for env in self.envs:
            env.close()
        self._save()
