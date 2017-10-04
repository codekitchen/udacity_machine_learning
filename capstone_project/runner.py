import sys
import time
from itertools import count

import gym
import gym_ple
from gym.wrappers import Monitor

import numpy as np
from scipy import misc

from dqn_agent import DQNAgent

EPISODES = 1000
BATCH_SIZE = 32

class ImagePreprocess:
    def __init__(self, env):
        self.env = env
        s = self.env.observation_space.shape
        self.observation_space = np.zeros([32, 32, 1])
    
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
        state = misc.imresize(state, (32, 32))
        state = (state * 255).astype(np.uint8)
        state = np.expand_dims(state, axis=2)
        return state
    
    def __getattr__(self, attr):
        return getattr(self.env, attr)

def main():
    agent_class = 'DQN'
    agent_name = sys.argv[2] if len(sys.argv) > 2 else 'Default'
    env_name = sys.argv[1] if len(sys.argv) > 1 else 'CartPole-v1'
    state_dir = "output/{}/{}/{}".format(agent_class, agent_name, env_name)
    monitor_dir = "{}/{}".format(state_dir, "video")
    env = gym.make(env_name)
    env = Monitor(env, monitor_dir, resume=True,
                  video_callable=lambda id: id % 20 == 0)
    im = len(env.observation_space.shape) == 3 
    if im:
        env = ImagePreprocess(env)
    agent = DQNAgent(env.observation_space.shape,
                     env.action_space.n, batch_size=BATCH_SIZE, state_dir=state_dir, image_input=im)

    for episode_num in count(1):
        state = env.reset()
        step = 0
        done = False
        score = 0
        st = time.time()
        while not done:
            step += 1
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.reward(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if env_name == 'PuckWorld-v0' and step > 1000:
                if hasattr(env, 'stats_recorder'):
                    # hack to work around stats_recorder blowing up
                    env.stats_recorder.done = True
                done = True
        per_step = (time.time() - st) / step
        print("episode: {}, time: {}, score: {}, e: {:.2} step: {}, perstep: {:.2}".format(
            episode_num, step, score, agent.epsilon, agent.step, per_step))

    env.close()


main()
