import gym
from gym.spaces import Space
from ray.rllib import MultiAgentEnv


class CorridorAgent:
    observation_space: Space = gym.spaces.Discrete(6)
    action_space: Space = gym.spaces.Discrete(2)

    def __init__(self, environment: MultiAgentEnv, number: int):
        self.id: str = 'corridor_agent_' + str(number)
        self._environment: MultiAgentEnv = environment
        self._current_position: int = 0
        self._current_reward: float = 0
        self._current_observation: int = None
        self._is_done: bool = False

    def compute_observation(self) -> int:
        self._current_observation = self._current_position
        return self._current_observation

    def compute_reward(self) -> float:
        if self._current_position >= self._environment.end_position:
            self._current_reward = 1
        else:
            self._current_reward = -0.1
        return self._current_reward

    def compute_is_done(self) -> bool:
        if self._current_position >= self._environment.end_position:
            self._is_done = True
        return self._is_done

    def compute_agent_information(self) -> dict:
        return {}

    def compute_action(self, action: int) -> None:
        if action == 0 and self._current_position > 0:
            self._current_position -= 1
        elif action == 1:
            self._current_position += 1
