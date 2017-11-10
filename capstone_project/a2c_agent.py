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
        self.env_count = 16
        self.t_steps = 5
        self.gamma = 0.99
        self.total_steps = 40e6
        self.starting_lr = 1e-3
        self.value_weight = 0.5
        self.entropy_weight = 0.01
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
                self.value_layer_val = self.value_layer[:, 0]
            with tf.variable_scope('state'):
                self._frames = tf.Variable(
                    0, trainable=False, name='frames', dtype=tf.int64)
                tf.summary.scalar('frames', self._frames)
                self.update_frames = tf.assign_add(self._frames,
                                                   tf.cast(tf.shape(self.input_layer)[0], tf.int64))
                lr_calc = self.starting_lr * \
                    (1.0 - (tf.cast(self._frames, tf.float64) / self.total_steps))
                # self.learning_rate = tf.maximum(tf.cast(0.0, tf.float64), lr_calc)
                self.learning_rate = tf.constant(self.starting_lr)
                tf.summary.scalar('learning_rate', self.learning_rate)
            with tf.variable_scope('training'):
                self.target_predict = tf.placeholder(
                    tf.int32, shape=[None], name='target_predict')
                self.target_value = tf.placeholder(
                    tf.float32, shape=[None], name='target_value')
                self.reward_diff = tf.placeholder(
                    tf.float32, shape=[None], name='reward_diff')
                mse_value = tf.reduce_mean(tf.squared_difference(
                    self.value_layer, self.target_value) / 2.)
                tf.summary.scalar('mse_value', mse_value)
                diff_predict = tf.reduce_mean(
                    self.reward_diff * tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.predict_layer, labels=self.target_predict))
                tf.summary.scalar('err_predict', diff_predict)
                a0 = self.predict_layer - \
                    tf.reduce_max(self.predict_layer, axis=1, keep_dims=True)
                ea0 = tf.exp(a0)
                z0 = tf.reduce_sum(ea0, axis=1, keep_dims=True)
                p0 = ea0 / z0
                # entropy = tf.reduce_mean(-tf.reduce_sum(self.softmax_predict * tf.log( self.softmax_predict + 1e-6), axis=1))  # adding 1e-6 to avoid DBZ
                entropy = tf.reduce_mean(tf.reduce_sum(
                    p0 * (tf.log(z0) - a0), axis=1))
                tf.summary.scalar('predict_entropy', entropy)
                trainer = tf.train.RMSPropOptimizer(
                    self.learning_rate, decay=0.99, epsilon=1e-5)
                loss = diff_predict + self.value_weight * \
                    mse_value - self.entropy_weight * entropy
                tf.summary.scalar('loss', loss)
                # self.train_op = trainer.minimize(loss, global_step=self._step)
                grads_and_vars = trainer.compute_gradients(loss)
                grads, vars = zip(*grads_and_vars)
                grads, _ = tf.clip_by_global_norm(grads, 0.5)
                grads_and_vars = list(zip(grads, vars))
                self.train_op = trainer.apply_gradients(
                    grads_and_vars, global_step=self._step)

            with tf.variable_scope('stats'):
                self.score_placeholder = tf.placeholder(
                    tf.float32, shape=[], name='score_input')
                score_1 = tf.Variable(0., trainable=False, name='score_1')
                tf.summary.scalar('score_1', score_1)
                score_100 = tf.Variable(0., trainable=False, name='score_100')
                tf.summary.scalar('score_100', score_100)
                score_1000 = tf.Variable(
                    0., trainable=False, name='score_1000')
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
        preds, values, _ = self.session.run(
            [self.softmax_predict, self.value_layer_val, self.update_frames],
            {self.input_layer: states})
        # noise = np.random.uniform(size=np.shape(preds))
        # return np.argmax(preds, axis=1)
        # return np.argmax(preds - np.log(-np.log(noise)), axis=1), values
        return [np.random.choice(self.action_count, p=pred) for pred in preds], values

    def value(self, states):
        """Returns predicted value for each given state"""
        vals = self.session.run(self.value_layer, {self.input_layer: states})
        return np.squeeze(vals, axis=1)

    def fit(self, states, actions, rewards, values):
        diff = rewards - values
        _, summary, step = self.session.run(
            [self.train_op, self.summary_data, self._step],
            {self.input_layer: states,
             self.target_value: rewards,
             self.target_predict: actions,
             self.reward_diff: diff})

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
            self.values = []

        @property
        def active(self):
            return not self.reset

        def prepare(self):
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.values.clear()
            if self.reset:
                self.current_state = self.env.reset()
                self.reset = False

        def async_step(self, action):
            if hasattr(self.env, 'async_step'):
                self.env.async_step(action)

        def step(self, action, value):
            self.states.append(self.current_state)
            self.actions.append(action)
            self.values.append(value)
            next_state, reward, self.done, self.reset = self.env.step(action)
            # Clip rewards to {-1, 0, 1}
            # This is done here so that the EnvRecorder wrapper can still report the
            # real score rather than the clipped score.
            reward = np.sign(reward)
            self.rewards.append(reward)
            self.current_state = next_state

    def run(self):
        self.start_time = time.time()
        run_envs = [self.RunEnv(env) for env in self.envs]
        last_save = 0
        while self.frames < self.total_steps:
            if self.frames - last_save > 100000:
                self._save()
                last_save = self.frames

            all_states = []
            all_actions = []
            all_rewards = []
            all_values = []

            for env in run_envs:
                env.prepare()

            for _ in range(self.t_steps):
                envs = [env for env in run_envs if env.active]
                if len(envs) == 0:
                    break
                states = [env.current_state for env in envs]
                actions, values = self.act(states)
                for env, action in zip(envs, actions):
                    env.async_step(action)
                for env, action, value in zip(envs, actions, values):
                    env.step(action, value)
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
                all_values += env.values
                all_rewards += total_rewards

            self.fit(all_states, all_actions, np.array(all_rewards), np.array(all_values))

        for env in self.envs:
            env.close()
        self._save()
