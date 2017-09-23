import sys
from itertools import count

import gym
from gym.wrappers import Monitor

from dqn_agent import DQNAgent

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
                  video_callable=lambda id: id % 10 == 0)
    agent = DQNAgent(env.observation_space.shape,
                     env.action_space.n, batch_size=BATCH_SIZE, state_dir=state_dir)

    for episode_num in count(1):
        state = env.reset()
        time = 0
        done = False
        score = 0
        while not done:
            time += 1
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.reward(state, action, reward, next_state, done)
            score += reward
            state = next_state
        print("episode: {}, time: {}, score: {}, e: {:.2}".format(
            episode_num, time, score, agent.epsilon))

    env.close()


main()
