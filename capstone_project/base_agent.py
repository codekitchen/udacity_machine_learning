"""Defines the base agent class"""

#pylint: disable=E1129

import tensorflow as tf

class BaseAgent:
    """BaseAgent shared functionality"""

    def __init__(self, first_env, state_dir):
        self.action_count = first_env.action_space.n
        self.state_shape = list(first_env.observation_space.shape)
        self.state_dir = state_dir
        self.image_input = hasattr(first_env, 'is_image') and first_env.is_image
        self._build_model()
        self._start_session()

    def _build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._step = tf.train.create_global_step()

    def _start_session(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            self.summary_data = tf.summary.merge_all()
            summary_dir = "{}/summary".format(self.state_dir)
            self.writer = tf.summary.FileWriter(summary_dir, self.graph)
            self.session.run(tf.global_variables_initializer())
            self.graph.finalize()
            checkpoint = tf.train.latest_checkpoint(self._snapshot_dir)
            if checkpoint:
                self._restore(checkpoint)
    
    def _restore(self, checkpoint):
        self.saver.restore(self.session, checkpoint)

    @property
    def step(self):
        return tf.train.global_step(
            self.session, tf.train.get_global_step(graph=self.session.graph))

    @property
    def _snapshot_dir(self):
        return "{}/snapshots".format(self.state_dir)

    def __del__(self):
        if self.session:
            self.session.close()
