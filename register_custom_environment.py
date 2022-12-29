from ray.tune import register_env
from simulation.multi_agent_corridor import MultiAgentCorridor


def multi_agent_corridor_env_creator(args):
    environment = MultiAgentCorridor(
        {
            'number_agents_start': 8,
            'max_step': 50,
            'max_number_agents': 10,
            'min_number_agents': 1,
        })
    return environment


register_env("multi_agent_corridor", multi_agent_corridor_env_creator)
