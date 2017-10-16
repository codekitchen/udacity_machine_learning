"""Implements the A2C Agent"""


class A2CAgent:
    """An Actor-Critic Advantage Network Agent"""

    def __init__(self, env_factory, state_dir):
        self.state_dir = state_dir