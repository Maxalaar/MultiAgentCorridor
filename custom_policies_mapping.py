import random

import gym
from ray.rllib.policy.policy import PolicySpec


def policies_dictionary():
    dictionary = {
        'policy1': PolicySpec(observation_space=gym.spaces.Discrete(6), action_space=gym.spaces.Discrete(2)),
        'policy2': PolicySpec(observation_space=gym.spaces.Discrete(6), action_space=gym.spaces.Discrete(2)),
    }

    return dictionary


def select_random_policy(agent_id, episode, worker, **kwargs):
    return random.choice([*policies_dictionary()])
