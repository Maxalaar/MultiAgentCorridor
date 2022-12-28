import random
import gym

from ray.rllib.policy.policy import PolicySpec

from simulation.corridor_agent import CorridorAgent


def policies_dictionary():
    observation_space = CorridorAgent.observation_space
    action_space = CorridorAgent.action_space
    dictionary = {
        'policy1': PolicySpec(observation_space=observation_space, action_space=action_space),
        'policy2': PolicySpec(observation_space=observation_space, action_space=action_space),
    }

    return dictionary


def select_random_policy(agent_id, episode, worker, **kwargs):
    return random.choice([*policies_dictionary()])
