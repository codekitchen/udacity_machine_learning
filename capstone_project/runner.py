"""Run one of the learning agents, for training or testing"""

import gym
from gym.wrappers import Monitor
import gym_ple # pylint: disable=unused-import

from env_recorder import EnvRecorder, EnvSharedState
from dqn_agent import DQNAgent
from a2c_agent import A2CAgent
from image import ImagePreprocess

# LunarLander-v2
# BipedalWalkerHardcore-v2
# CarRacing-v0
# PuckWorld-v0
# SpaceInvadersNoFrameskip-v4

def vid_schedule(cap=1000):
    def _sched(episode_id):
        if episode_id < cap:
            return int(round(episode_id ** (1. / 3))) ** 3 == episode_id
        return episode_id % cap == 0
    return _sched

def main():
    """Read args, construct agent, and run it"""

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--agent', help='agent class', default='DQN')
    parser.add_argument('--env', help='env name', default='CartPole-v1')
    parser.add_argument('--name', help='agent name (for saving info)', default='Default')
    args = parser.parse_args()

    agent_class_name = args.agent.upper()
    agent_class = {
        'DQN': DQNAgent,
        'A2C': A2CAgent,
    }[agent_class_name]

    monitor_cap = {
        'DQN': 1000,
        'A2C': 166,
    }[agent_class_name]

    state_dir = "output/{}/{}/{}".format(agent_class_name, args.name, args.env)
    monitor_dir = "{}/{}".format(state_dir, "video")

    env_count = 0
    env_shared_state = EnvSharedState()
    def _make_env():
        env = gym.make(args.env)
        nonlocal env_count
        env.seed(env_count)
        # add a Monitor to the first env, to get video output
        if env_count == 0:
            env = Monitor(env, monitor_dir, resume=True, video_callable=vid_schedule(monitor_cap))
        # simple heuristic that should work to detect envs with images as input
        is_image = len(env.observation_space.shape) == 3
        if is_image:
            env = ImagePreprocess(args.env, env)
        env_count += 1
        return EnvRecorder(env, env_shared_state)

    agent = agent_class(_make_env, state_dir=state_dir)
    agent.run()

main()
