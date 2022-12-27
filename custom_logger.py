import os
import ray
from datetime import datetime
from pathlib import Path


def custom_logger_creator(config: dict):
    from ray.tune.logger import UnifiedLogger
    path_results: Path = Path('./results/')
    if not path_results.is_dir():
        os.mkdir(path_results)

    time_string = datetime.today().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    name_directory_experiment: str = "{}_{}".format("test", time_string)
    path_experiment: Path = path_results.joinpath(name_directory_experiment)
    if not path_experiment.is_dir():
        os.mkdir(path_experiment)

    ray._private.utils.try_to_create_directory(path_experiment)
    return UnifiedLogger(config, path_experiment)
