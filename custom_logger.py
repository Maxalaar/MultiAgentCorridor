import os
import ray
from datetime import datetime
from pathlib import Path
import json
from ray.rllib.algorithms.algorithm import AlgorithmConfig
from ray.tune.logger import pretty_print

PATH_EXPERIMENT: Path = None
def custom_logger_creator(algorithm_config: AlgorithmConfig):
    # create 'results' directory
    from ray.tune.logger import UnifiedLogger
    path_results: Path = Path('./results/')
    if not path_results.is_dir():
        os.mkdir(path_results)

    # create path experiment
    time_string = datetime.today().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    name_directory_experiment: str = str(algorithm_config['env']) + '_' + str(time_string)
    path_experiment: Path = path_results.joinpath(name_directory_experiment)
    PATH_EXPERIMENT = path_experiment

    # create directory of path experiment
    if not path_experiment.is_dir():
        os.mkdir(path_experiment)
    ray._private.utils.try_to_create_directory(path_experiment)

    # save algorithm_config in text file
    path_algorithm_config: Path = path_experiment.joinpath('algorithm_config.txt')
    with open(path_algorithm_config, 'w') as algorithm_config_file:
        algorithm_config_file.write(pretty_print(algorithm_config.to_dict()))

    return UnifiedLogger(algorithm_config, path_experiment)
