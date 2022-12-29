import random
import gym

from ray.rllib.policy.policy import PolicySpec

from environment.corridor_agent import CorridorAgent


def policies_dictionary():
    observation_space = CorridorAgent.observation_space
    action_space = CorridorAgent.action_space
    configuration = {
        'model': {
            'custom_model': 'minimal_model',
        },
    }
    dictionary = {
        'policy1': PolicySpec(observation_space=observation_space, action_space=action_space, config=configuration),
        'policy2': PolicySpec(observation_space=observation_space, action_space=action_space, config=configuration),
    }

    return dictionary


def select_random_policy(agent_id, episode, worker, **kwargs):
    return random.choice([*policies_dictionary()])
