import sys
import time
from itertools import count

import gym
import gym_ple
from gym.wrappers import Monitor

from dqn_agent import DQNAgent
from image import ImagePreprocess

EPISODES = 1000
BATCH_SIZE = 32

def main():
    agent_class = 'DQN'
    agent_name = sys.argv[2] if len(sys.argv) > 2 else 'Default'
    env_name = sys.argv[1] if len(sys.argv) > 1 else 'CartPole-v1'
    state_dir = "output/{}/{}/{}".format(agent_class, agent_name, env_name)
    monitor_dir = "{}/{}".format(state_dir, "video")
    env = gym.make(env_name)
    env = Monitor(env, monitor_dir, resume=True,
                  video_callable=lambda id: id % 20 == 0)
    # simple heuristic that should work to detect envs with images as input
    im = len(env.observation_space.shape) == 3 
    if im:
        env = ImagePreprocess(env_name, env)
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
        print("episode: {}, time: {}, score: {:.2f}, e: {:.2} step: {}, perstep: {:.2}".format(
            episode_num, step, score, agent.epsilon, agent.step, per_step))

    env.close()


main()
