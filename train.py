import ray

from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from custom_logger import custom_logger_creator

ppo_classic_config = PPOConfig()\
    .environment("Taxi-v3")\
    .debugging(logger_creator=custom_logger_creator)

if __name__ == '__main__':
    ray.init(local_mode=True)

    algorithm_config: AlgorithmConfig = ppo_classic_config

    algorithm: Algorithm = algorithm_config.build()

    for i in range(10):
        algorithm.train()

