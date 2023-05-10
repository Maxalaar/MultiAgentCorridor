from typing import Dict, Tuple, Union
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID


class CustomCallbacks(DefaultCallbacks):
    def on_episode_created(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            env_index: int,
            episode: Union[Episode, EpisodeV2],
            **kwargs,
    ) -> None:
        pass
