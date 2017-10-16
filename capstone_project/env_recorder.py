import time


class EnvRecorder:
    def __init__(self, env):
        self.env = env
        self.episode = 0
        self.frame = 0
        self.ep_frame = 0
        self.ep_score = 0
        self.reports = 0
        self.done = False
        self.start_time = time.time()
        self.info_cb = lambda *a: []

    def reset(self):
        state = self.env.reset()
        self.episode += 1
        self.ep_frame = 0
        self.ep_score = 0
        self.done = False
        self.start_time = time.time()
        return state

    def step(self, action):
        self.frame += 1
        self.ep_frame += 1
        res = self.env.step(action)
        state, reward, done, _ = res
        reset = done
        self.ep_score += reward
        if self.env.spec.id == 'PuckWorld-v0' and self.ep_frame >= 2000:
            if hasattr(self.env, 'stats_recorder'):
                # hack to work around stats_recorder blowing up
                self.env.stats_recorder.done = True
            reset = True
        if reset:
            self._finish()
        return state, reward, done, reset

    def _finish(self):
        per_frame = (time.time() - self.start_time) / self.ep_frame
        stats = [
            ('episode', self.episode),
            ('time', self.ep_frame),
            ('score', '{:.2f}'.format(self.ep_score)),
            ('perframe', '{:.3f}'.format(per_frame)),
        ] + self.info_cb(self)
        print(", ".join(["{}: {}".format(k, v) for k, v in stats]))
        self.reports += 1

    def __getattr__(self, attr):
        return getattr(self.env, attr)
