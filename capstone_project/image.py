import numpy as np
from scipy import misc

SIZES = {
    'PuckWorld-v0': (32, 32),
}

class ImagePreprocess:
    def __init__(self, env_name, env):
        self.env = env
        self.imsize = SIZES[env_name] if env_name in SIZES else (84, 84)
        self.observation_space = np.zeros(list(self.imsize) + [1])

    def reset(self):
        state = self.env.reset()
        return self.process(state)

    def step(self, action):
        res = list(self.env.step(action))
        res[0] = self.process(res[0])
        return res

    def process(self, state):
        # Simple luminance extraction by taking the mean of all 3 channels
        state = np.mean(state, axis=2)
        state = misc.imresize(state, self.imsize)
        state = (state * 255).astype(np.uint8)
        state = np.expand_dims(state, axis=2)
        return state
    
    def __getattr__(self, attr):
        return getattr(self.env, attr)