import gym
from ray.rllib import MultiAgentEnv

from simulation.corridor_agent import CorridorAgent


class MultiAgentCorridor(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.number_agents = config['number_agents']
        self.end_position = config['corridor_length']

        self._spaces_in_preferred_format = True
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

        self.action_space = gym.spaces.Dict()
        self.observation_space = gym.spaces.Dict()

        self._agent_ids = set()
        self.agents_list: list[CorridorAgent] = []
        self.agents_dictionary: dict[str:CorridorAgent] = {}

        for i in range(self.number_agents):
            agent = CorridorAgent(self, i)
            self._agent_ids.add(agent.id)
            self.agents_list.append(agent)
            self.agents_dictionary[agent.id] = agent

            self.observation_space[agent.id] = agent.observation_space
            self.action_space[agent.id] = agent.action_space

    def reset(self):
        for agent in self.agents_list:
            agent.reset()

        observations_dictionary = {}
        for agent in self.agents_list:
            observations_dictionary[agent.id] = agent.compute_observation()

        return observations_dictionary

    def step(self, action_dictionary: dict[str:int]):
        for agent_id, action in action_dictionary.items():
            self.agents_dictionary[agent_id].compute_action(action)

        all_agents_is_done: bool = True
        observations_dictionary = {}
        reward_dictionary = {}
        is_done_dictionary = {}
        information_dictionary = {}

        for agent_id, _ in action_dictionary.items():
            agent = self.agents_dictionary[agent_id]
            observations_dictionary[agent.id] = agent.compute_observation()
            reward_dictionary[agent.id] = agent.compute_reward()
            is_done_dictionary[agent.id] = agent.compute_is_done()
            information_dictionary[agent.id] = agent.compute_information()
            if all_agents_is_done and not agent.compute_is_done():
                all_agents_is_done = False
        is_done_dictionary['__all__'] = all_agents_is_done

        return observations_dictionary, reward_dictionary, is_done_dictionary, information_dictionary
