import numpy as np
import cv2

SIZES = {
    'PuckWorld-v0': (32, 32),
}
GRAY = {
    #'PuckWorld-v0': False,
}


class ImagePreprocess:
    """gym env wrapper to handle image input

    This handles both preprocessing the image (rescaling and converting to
    grayscale) as well adding a history dimension, turning the input state
    into the last `history` frames by adding more channels.

    History is important when using images as input because it's the only way
    for the agent to determine velocity. This is done by adding more
    channels, rather than adding another dimension, because adding another
    dimension would require changing the CNNs.
    """

    is_image = True

    def __init__(self, env_name, env, history=4):
        self.env = env
        self.imsize = SIZES.get(env_name, (84, 84))
        self.gray = GRAY.get(env_name, True)
        self.chans = 1 if self.gray else env.observation_space.shape[2]
        self.history = history
        self.observation_space = np.zeros(
            list(self.imsize) + [self.chans * self.history], dtype=np.uint8)
        self.state = np.zeros_like(self.observation_space)

    def reset(self):
        first_state = self._process(self.env.reset())
        # "clear" all the history to this initial state
        for _ in range(self.history):
          self._update_state(first_state)
        return self.state

    def step(self, action):
        res = list(self.env.step(action))
        res[0] = self._update_state(self._process(res[0]))
        return res

    def _process(self, state):
        if self.gray:
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.imsize, interpolation=cv2.INTER_AREA)
        if self.gray:
            state = np.expand_dims(state, axis=2)
        return state

    def _update_state(self, processed_state):
        self.state = np.roll(self.state, shift=self.chans, axis=2)
        self.state[:, :, 0:self.chans] = processed_state
        return self.state

    def __getattr__(self, attr):
        return getattr(self.env, attr)
