import ray

from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import AlgorithmConfig, Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

if __name__ == '__main__':
    ray.init(local_mode=True)

    algorithm_config: AlgorithmConfig = PPOConfig().environment("Taxi-v3")

    algorithm: Algorithm = algorithm_config.build()

    for i in range(10):
        algorithm.train()

