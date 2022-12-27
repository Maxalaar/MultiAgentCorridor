import ray
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole

import register_custom_environment
from pathlib import Path
from custom_logger import custom_logger_creator
from custom_policies_mapping import policies_dictionary, select_random_policy
from simulation.multi_agent_corridor import MultiAgentCorridor

ppo_classic_config = (
    PPOConfig()
    .framework('torch')

    .environment('multi_agent_corridor', disable_env_checking=True)
    # .environment(MultiAgentCorridor, env_config={'number_agents': 2, 'corridor_length': 5}, disable_env_checking=True)
    # .environment(MultiAgentCartPole, env_config={"num_agents": 2})

    .debugging(logger_creator=custom_logger_creator)
    .multi_agent(policies=policies_dictionary(), policy_mapping_fn=select_random_policy)
)

if __name__ == '__main__':
    ray.init(local_mode=True)

    algorithm_config: AlgorithmConfig = ppo_classic_config

    algorithm: Algorithm = algorithm_config.build()

    for i in range(1):
        algorithm.train()
        algorithm.save(Path(algorithm.logdir + '/checkpoint/'))
