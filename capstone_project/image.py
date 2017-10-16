import numpy as np
from scipy import misc

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
        self.state = np.zeros_like(self.observation_space)
        first_state = self.env.reset()
        return self.process(first_state)

    def step(self, action):
        res = list(self.env.step(action))
        res[0] = self.process(res[0])
        return res

    def process(self, state):
        if self.gray:
            # Simple luminance extraction by taking the mean of all 3 channels
            state = np.mean(state, axis=2)
        state = misc.imresize(state, self.imsize)
        state = (state * 255).astype(np.uint8)
        if self.gray:
            state = np.expand_dims(state, axis=2)
        self.state = np.roll(self.state, shift=self.chans, axis=2)
        self.state[:, :, 0:self.chans] = state
        return self.state

    def __getattr__(self, attr):
        return getattr(self.env, attr)
