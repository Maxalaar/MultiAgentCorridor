import gym
import random
from ray.rllib import MultiAgentEnv

from simulation.corridor_agent import CorridorAgent


class MultiAgentCorridor(MultiAgentEnv):
    def __init__(self, environment_configuration: dict):
        super().__init__()
        self.number_agents_start: int = environment_configuration['number_agents_start']
        self.max_number_agents: int = environment_configuration['max_number_agents']
        self.min_number_agents: int = environment_configuration['min_number_agents']
        self.max_step: int = environment_configuration['max_step']
        self.end_position: int = 5

        self._spaces_in_preferred_format: bool = True
        self._obs_space_in_preferred_format: bool = True
        self._action_space_in_preferred_format: bool = True

        self.current_step: int = None
        self.next_agent_id: set = None
        self.agents_dictionary: dict = None
        self.agents_list: list = None

        self.observations_dictionary: dict = None
        self.agents_information_dictionary: dict = None
        self.is_done_dictionary: dict = None
        self.rewards_dictionary: dict = None

    def reset(self):
        self.action_space = gym.spaces.Dict()
        self.observation_space = gym.spaces.Dict()

        self.current_step = 0
        self.next_agent_id: int = 0
        self._agent_ids: dict = set()
        self.agents_list: list[CorridorAgent] = []
        self.agents_dictionary: dict[str:CorridorAgent] = {}
        self.clear_information_dictionaries()

        for _ in range(self.number_agents_start):
            self.add_agent()

        return self.observations_dictionary

    def step(self, action_dictionary: dict):
        self.clear_information_dictionaries()
        self.current_step += 1

        for agent_id, action in action_dictionary.items():
            self.agents_dictionary[agent_id].compute_action(action)

        for agent_id, _ in action_dictionary.items():
            self.update_information_dictionaries(agent_id)

        if random.random() < 0.25 and len(self._agent_ids) < self.max_number_agents:
            self.add_agent()

        if random.random() < 0.25 and len(self._agent_ids) > self.min_number_agents:
            agent_id = random.choice(list(self._agent_ids))
            self.remove_agent(agent_id)

        self.compute_simulation_is_done()

        return self.observations_dictionary, self.rewards_dictionary, self.is_done_dictionary, self.agents_information_dictionary

    def clear_information_dictionaries(self):
        self.observations_dictionary = {}
        self.rewards_dictionary = {}
        self.is_done_dictionary = {}
        self.agents_information_dictionary = {}

    def update_information_dictionaries(self, agent_id: str):
        agent = self.agents_dictionary[agent_id]
        self.observations_dictionary[agent.id] = agent.compute_observation()
        self.rewards_dictionary[agent.id] = agent.compute_reward()
        self.is_done_dictionary[agent.id] = agent.compute_is_done()
        self.agents_information_dictionary[agent.id] = agent.compute_agent_information()

    def compute_simulation_is_done(self):
        simulation_is_done = True

        # if an agent still has to act, the simulation is not stopped
        for agent in self.agents_list:
            if not agent.compute_is_done():
                simulation_is_done = False
                break

        # if the current step is equal or higher than the max step, the simulation is stopped
        if self.current_step >= self.max_step:
            simulation_is_done = True

        self.is_done_dictionary['__all__'] = simulation_is_done  # False

    def add_agent(self):
        agent = CorridorAgent(self, self.next_agent_id)
        self.next_agent_id += 1
        self._agent_ids.add(agent.id)
        self.agents_list.append(agent)
        self.agents_dictionary[agent.id] = agent

        self.observation_space[agent.id] = agent.observation_space
        self.action_space[agent.id] = agent.action_space

        self.update_information_dictionaries(agent.id)

    def remove_agent(self, agent_id: str):
        assert agent_id in self._agent_ids != True, 'Attempt to delete an agent that does not exist'
        agent: CorridorAgent = self.agents_dictionary[agent_id]

        self._agent_ids.remove(agent_id)
        self.agents_list.remove(agent)
        del self.agents_dictionary[agent_id]
        del self.observation_space.spaces[agent_id]
        del self.action_space.spaces[agent_id]

        if agent_id in self.is_done_dictionary.keys():
            self.is_done_dictionary[agent.id] = True
