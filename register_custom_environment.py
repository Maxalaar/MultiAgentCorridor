from ray.tune import register_env
from simulation.multi_agent_corridor import MultiAgentCorridor


def multi_agent_corridor_env_creator(args):
    environment = MultiAgentCorridor({'number_agents': 2, 'corridor_length': 5})
    return environment


register_env("multi_agent_corridor", multi_agent_corridor_env_creator)
