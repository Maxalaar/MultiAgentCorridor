import ray

from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from custom_logger import custom_logger_creator

import register_custom_environment

ppo_classic_config = PPOConfig()\
    .environment('multi_agent_corridor')\
    .debugging(logger_creator=custom_logger_creator)

if __name__ == '__main__':
    ray.init(local_mode=True)

    algorithm_config: AlgorithmConfig = ppo_classic_config

    algorithm: Algorithm = algorithm_config.build()

    for i in range(10):
        algorithm.train()

