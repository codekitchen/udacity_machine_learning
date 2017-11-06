"""Implements a gym env in a subprocess communicating over pipes"""

import gym
from multiprocessing import Process, Pipe

def worker(pipe, env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    while True:
        cmd, args = pipe.recv()
        if cmd == 'step':
            res = env.step(args)
            pipe.send(res)
        elif cmd == 'reset':
            res = env.reset()
            pipe.send(res)
        elif cmd == 'info':
            pipe.send((env.action_space, env.observation_space, env.spec))
        elif cmd == 'close':
            env.close()
            break
        else:
            raise NotImplementedError

class SubProcEnv:
    def __init__(self, env_name, seed=0):
        self.pipe, child_pipe = Pipe()
        self.proc = Process(target=worker, args=(child_pipe, env_name, seed))
        self.proc.start()
        self._async_stepped = False
        self.closed = False

        self.pipe.send(('info', None))
        self.action_space, self.observation_space, self.spec = self.pipe.recv()

    def async_step(self, action):
        self.pipe.send(('step', action))
        self._async_stepped = True

    def step(self, action):
        if not self._async_stepped:
            print('going sync')
            self.async_step(action)
        self._async_stepped = False
        return self.pipe.recv()

    def reset(self):
        self.pipe.send(('reset', None))
        return self.pipe.recv()

    def close(self):
        self.pipe.send(('close', None))
        self.proc.join()
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.proc.terminate()
