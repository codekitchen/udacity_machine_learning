"""Implements the A2C Agent"""

#pylint: disable=E1129

import tensorflow as tf

from base_agent import BaseAgent
from layers import agent_network, fully_connected


class A2CAgent(BaseAgent):
    """An Actor-Critic Advantage Network Agent"""

    def __init__(self, env_factory, state_dir):
        self.env_count = 6
        self.t_steps = 5
        self.envs = []
        for i in range(self.env_count):
            env = env_factory()
            self.envs.append(env)
            env.a2c_id = i
            env.info_cb = self.status
        super().__init__(self.envs[0], state_dir)
        print("A2C input shape ", self.state_shape)

    def _build_model(self):
        super()._build_model()
        with self.graph.as_default():
            input_layer, final_layer, predict_layer = agent_network(
                state_shape=self.state_shape,
                image_input=self.image_input,
                action_count=self.action_count)
            self.input_layer = input_layer
            # TODO: add noise?
            self.predict_layer = tf.argmax(
                predict_layer, axis=1, name='action')
            self.value_layer = fully_connected(
                final_layer, 1, activation=None, name="value")

    def status(self, env):
        return [('env', env.a2c_id)]

    def act(self, states):
        """Returns action and predicted value for each given state"""
        return self.session.run([self.predict_layer, self.value_layer], {
            self.input_layer: states})

    def run(self):
        states = [env.reset() for env in self.envs]
        while True:
            for _ in range(self.t_steps):
                actions, _ = self.act(states)
                results = [env.step(action)
                           for env, action in zip(self.envs, actions)]
                next_states, rewards, dones, resets = zip(*results)
                next_states = list(next_states)

                # reset any 'done' envs
                for i, (env, reset) in enumerate(zip(self.envs, resets)):
                    if reset:
                        next_states[i] = env.reset()
