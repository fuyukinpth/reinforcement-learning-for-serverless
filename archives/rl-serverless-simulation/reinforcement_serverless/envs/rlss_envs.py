import os
import time
import threading
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from utils.poisson_simulation import Poisson

'''
Define an index corresponding to the action that changes the container's state:
    - Destination state is changed from the original state: 1
    - Source state is changed to another state: -1
    - State is not changed: 0
'''
class Actions:
    destination_state = 1
    source_state = -1
    no_change_state = 0

'''    
Defines symbols in a state machine:
    - N  = Null
    - L0 = Cold
    - L1 = Warm Disk
    - L2 = Warm CPU
    - A  = Active
'''
class States:
    n = "Null"
    l_0 = "Cold"
    l_1 = "Warm Disk"
    l_2 = "Warm CPU"
    a = "Active"
    
'''
Define cases where state changes can occur:
    N <-> L0 <-> L1 <-> L2 <-> A
'''    
class Transitions:
    trans_0 = np.array([-1, 1, 0, 0, 0])   # N -> L0
    trans_1 = np.array([1, -1, 0, 0, 0])   # L0 -> N
    trans_2 = np.array([0, -1, 1, 0, 0])   # L0 -> L1
    trans_3 = np.array([0, 1, -1, 0, 0])   # L1 -> L0
    trans_4 = np.array([0, 0, -1, 1, 0])   # L1 -> L2
    trans_5 = np.array([0, 0, 1, -1, 0])   # L2 -> L1
    trans_6 = np.array([1, -1, -1, 1, 0])  # L0 -> N and L1 -> L2
    trans_7 = np.array([-1, 1, 1, -1, 0])  # N -> L0 and L2 -> L1
    trans_8 = np.array([1, -1, 1, -1, 0])  # L0 -> N and L2 -> L1
    trans_9 = np.array([-1, 1, -1, 1, 0])  # N -> L0 and L1 -> L2


class ServerlessEnv(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None, size=4):
        super(ServerlessEnv, self).__init__()
        '''
        Define environment parameters
        '''
        self.size = size  # The number of services
        self.num_states = 5  # The number of states in a container's lifecycle (N, L0, L1, L2, A)
        self.num_resources = 3  # The number of resource parameters (RAM, GPU, CPU)
        self.min_container = 16
        self.max_container = 256
        
        self.timeout = 10  # Set timeout value = 10s
        self.container_lifetime = 86400  # Set lifetime of a container = 1 day
        self.limited_ram = 64  # Set limited amount of RAM (server) = 64GB
        self.limited_requests = 1280  # Set the limit number of requests that can exist in the system = 1280
        self.average_requests = 640  # Set the average incoming requests per hour = 640

        '''
        Define observations (state space)
        '''
        self.observation_space = spaces.Dict({
            "execution_times":   spaces.Box(low=1, high=10, shape=(self.size, 1), dtype=np.int16),
            "request_quantity":  spaces.Box(low=0, high=self.limited_requests, shape=(self.size, 1), dtype=np.int16),
            "remained_resource": spaces.Box(low=0, high=self.limited_ram, shape=(self.num_resources, 1), dtype=np.int16),
            "container_traffic": spaces.Box(low=0, high=self.max_container, shape=(self.size, self.num_states), dtype=np.int16),
        })
        
        '''
        Define action space containing two matrices by combining them into a Tuple space
        '''
        self._action_coefficient = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int16)
        self._action_unit = spaces.Box(low=-1, high=1, shape=(self.size, self.num_states), dtype=np.int16)
        self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))
        
        # Set the main diagonal elements of _action_coefficient to be in the range [0, self.max_container] 
        np.fill_diagonal(self._action_coefficient.low, 0)
        np.fill_diagonal(self._action_coefficient.high, self.max_container)
        
        # Set the last column of the _action_unit to be always zero and the sum of the elements in a row of the _action_unit = 0 using _get_units()
        # self._action_unit.low[:, -1] = 0
        # self._action_unit.high[:, -1] = 0
        self._get_units()
        
        '''
        Initialize the state and other variables
        '''
        self.current_time = 0  # Start at time 0
        self.transition_ram = 0  # Set an initial value
        self.transition_time = 0  # Set an initial value
        self.transition_power = 0  # Set an initial value
        self.processed_requests = 0  # Set an initial value
        self._pending_request = np.zeros((self.size, 1), dtype=np.int16)  # Set an initial value
        self._ram_required_matrix = np.array([0, 0, 0, 0.9, 2])  # Set the required RAM each state
        self._action_matrix = np.zeros((self.size, self.num_states), dtype=np.int16)  # Set an initial value
        self._exectime_matrix = np.random.randint(2, 16, size=(self.size, 1))
        self._request_matrix = np.zeros((self.size, 1), dtype=np.int32)
        self._resource_matrix = np.ones((self.num_resources, 1), dtype=np.int16)
        self._resource_matrix[0, 0] = self.limited_ram
        self._container_matrix = np.hstack((
            np.random.randint(self.min_container, self.max_container, size=(self.size, 1)),  # Initially the containers are in Null state
            np.zeros((self.size, self.num_states-1), dtype=np.int16)
        ))
        self._is_changing_matrix = np.zeros((self.size, self.num_states), dtype=np.int16)  # Set an initial value
        
        '''
        Create and start thread objects for parallel execution
        '''
        self.thread_time = threading.Thread(target=self._get_time)
        self.thread_request = threading.Thread(target=self._get_request)
        self.thread_pending = threading.Thread(target=self._get_pending)
        self.thread_system = threading.Thread(target=self._get_system)
        
        self.thread_time.start()
        self.thread_request.start()
        self.thread_pending.start()
        self.thread_system.start()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_units(self):
        '''
        Define a function to create a random action unit matrix
        '''
        # while True:
        #     random_matrix = np.random.randint(-1, 2, size=(self.size, self.num_states-1))  # Generate a random matrix
        #     row_sums = np.sum(random_matrix, axis=1)  # Ensure the row sum is 0
        #     if np.all(row_sums == 0):  # If row sum is 0, assign the matrix to _action_unit
        #         self._action_unit.low[:, :-1] = random_matrix
        #         self._action_unit.high[:, :-1] = random_matrix
        #         break     
        array_set = [getattr(Transitions, attr) for attr in dir(Transitions) if not attr.startswith("__")]
        random_matrix = np.array([array_set[np.random.randint(0, len(array_set))] for _ in range(self.size)])
        self._action_unit.low[:, :] = random_matrix
        self._action_unit.high[:, :] = random_matrix
        
    def _get_reward(self):
        '''
        Define reward calculation function
        '''
        # a, b = 100, 100  # <-- Customize the coefficients a and b here
        # info = self._get_info()
        # reward = a * info["system_profit"] - b * info["energy_consumption"] - info["penalty_delay"] - info["penalty_abandone"]
            
        reward = self.processed_requests
        return reward  
        
    def _get_constraints(self, action):
        '''
        Define a function that checks constraint conditions (action masking)
        '''
        temp_matrix = self._container_matrix
        temp_matrix += action[0] @ action[1]
        if (np.any(temp_matrix < 0)): 
            action = self.action_space.sample()  # Make sure there are no negative values in _container_matrix
            return False
        
        temp_matrix *= self._ram_required_matrix
        if (np.sum(temp_matrix) > self._resource_matrix[0, 0]):
            return False  # Make sure there is no RAM overflow
        
        return True
    
    def _get_time(self):
        '''
        Define a function that counts the time in the system
        '''
        while True:
            time.sleep(1)  # Wait for 1 second
            self.current_time += 1
    
    def _get_request(self):
        '''
        Define a function that receives an amount of incoming requests every Δt based on Poisson distribution
        '''
        request_per_day = self.average_requests * 24
        requests_per_second = self.average_requests / 3600
        time_duration =  self.container_lifetime
        poisson_requests = Poisson(rate=requests_per_second, time_duration=time_duration)
        incoming_request_times = np.zeros((self.size, request_per_day))
        
        for i in range(self.size):
            num_events, event_times, inter_arrival_times = poisson_requests.generate_poisson_events()
            if (num_events > request_per_day):
                temp_matrix = np.zeros((self.size, num_events))
                row_index = 0
                col_index = 0
                temp_matrix[row_index:row_index + incoming_request_times.shape[0], col_index:col_index + incoming_request_times.shape[1]] = incoming_request_times
                incoming_request_times = temp_matrix
            
            incoming_request_times[i, :num_events] = event_times
        
        while(True):
            unique_elements = np.unique(incoming_request_times)
            sorted_elements = np.sort(unique_elements)
            incoming_timeline = sorted_elements[sorted_elements != 0]

            for i in range (0, len(incoming_timeline)):
                indices = np.where(incoming_request_times == incoming_timeline[i])  # Find the indices where the element occurs in incoming_request_times
                time.sleep(incoming_timeline[i] - incoming_timeline[i-1] if i != 0 else 0)  # Wait for the delay before adding to _request_matrix
                for idx in indices[0]:
                    self._request_matrix[idx] += 1
        
    def _get_pending(self):
        '''
        Define a function that eliminates request failure due to timeout
        '''
        while True:
            time.sleep(self.timeout)  # Wait for 'timeout' seconds
            self._request_matrix -= self._pending_request
            self._pending_request = np.zeros((self.size, 1), dtype=np.int16)
    
    def _get_transition_time(self, action):
        '''
        Define a function that calculates time consumed when switching states
        '''
        pass
    
    def _get_transition_power(self, action):
        '''
        Define a function that calculates power consumed when switching states
        '''
        pass
    
    def _get_transition_ram(self, action):
        '''
        Define a function that calculates RAM consumed when switching states
        '''
        pass
    
    def _get_system(self, action):
        # temp_matrix = self._container_matrix
            
        # for service in range(self.size):
        #     if (self._container_matrix[service, 4] >= self._request_matrix[service, 0]):
        #         self._get_transition_power(action)
        #         self._get_transition_time(action)
        #         self._container_matrix[service, 4] -= self._request_matrix[service, 0]
        #         self._request_matrix[service, 0] = 0
        #     else:
        #         self._get_transition_power(action)
        #         self._get_transition_time(action)
        #         self._request_matrix[service, 0] -= self._container_matrix[service, 4]
        #         self._container_matrix[service, 4] = 0
            
        # time.sleep(0.05)  # Time to switchs state from L2 to A
        # time_matrix = np.sort(self._exectime_matrix, axis=0)
            
        # for i in range(self.size):
        #     time.sleep(time_matrix[i, 0])  # Execution time
        #     for j in range(self.size):
        #         if (self._exectime_matrix[j, 0] == time_matrix[i, 0]):
        #             self._get_transition_power(action)
        #             self._get_transition_time(action)
        #             self._container_matrix[j, 4] += (temp_matrix[j, 4] - self._container_matrix[j, 4])  # Containers at WarmCPU switches to Active to process requests
        #             time.sleep(0)  # Time to switchs state from A back to L2
        #     time_matrix -= time_matrix[i, 0] 
        
        pass
            
    def _get_obs(self):
        '''
        Define a function that returns the values of observation
        '''
        return {
            "execution_times": self._exectime_matrix,
            "request_quantity": self._request_matrix,
            "remained_resource": self._resource_matrix,
            "container_traffic": self._container_matrix,
        }

    def _get_info(self):
        '''
        Defines a function that returns system evaluation parameters
        '''
        profit = 0
        ram_used_per_service = 0
        cost_per_unit = 0
        
        for service in range(self.size):
            ram_used_per_service += np.sum(self._container_matrix[service] * self._ram_required_matrix) + self.transition_ram
            profit_per_service = self.average_requests * ram_used_per_service * \
                                 self._exectime_matrix[service] * cost_per_unit
            profit += profit_per_service[0]
        
        # abandone_rate = (0.058 * self.current_time) if self.current_time > 2 else 0
        # time_temp = self.current_time if self.current_time > 2 else 0
        # alpha_temp = 0.05 * profit
        # beta_temp = 0.34
        
        abandone_rate = 0
        time_temp = 0
        alpha_temp = 0
        beta_temp = 0
        
        return {
            "energy_consumption": 100,
            "penalty_delay": alpha_temp + beta_temp * time_temp,
            "penalty_abandone": profit * abandone_rate,
            "system_profit": profit,
        }
    
    def reset(self, seed=None, options=None):
        '''
        Initialize the environment
        '''
        super().reset(seed=seed) # We need the following line to seed self.np_random
        
        self.current_time = 0  # Start at time 0
        self.transition_ram = 0  # Set an initial value
        self.transition_time = 0  # Set an initial value
        self.transition_power = 0  # Set an initial value
        self.processed_requests = 0  # Set an initial value
        self._pending_request = np.zeros((self.size, 1), dtype=np.int16)  # Set an initial value
        self._exectime_matrix = np.random.randint(2, 16, size=(self.size, 1))
        self._request_matrix = np.zeros((self.size, 1), dtype=np.int32)
        self._resource_matrix = np.ones((self.num_resources, 1), dtype=np.int16)
        self._resource_matrix[0, 0] = self.limited_ram
        self._container_matrix = np.hstack((
            np.random.randint(self.min_container, self.max_container, size=(self.size, 1)),  # Initially the containers are in Null state
            np.zeros((self.size, self.num_states-1), dtype=np.int16)
        ))
        self._is_changing_matrix = np.zeros((self.size, self.num_states), dtype=np.int16)  # Set an initial value
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        '''
        Apply action masking to check the validity of status updates
        '''
        if (self._get_constraints):
            self._action_matrix = action[0] @ action[1]
            self._container_matrix += self._action_matrix
            # Add editing algorithm (if any)
        else: pass
      
        temp_matrix = self._container_matrix * self._ram_required_matrix
        terminated = self.current_time >= self.container_lifetime
        truncated = False
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        '''
        Implement a visualization method
        '''
        info = self._get_info()
        reward = self._get_reward()
        print("SYSTEM EVALUATION PARAMETERS:")
        print("- Energy Consumption  : {:.2f}J".format(info["energy_consumption"]))
        print("- Delay Penalty       : {:.2f}".format(info["penalty_delay"]))
        print("- Abandone Penalty    : {:.2f}".format(info["penalty_abandone"]))
        print("- Profit              : {:.2f}".format(info["system_profit"]))
        print("- Reward              : {:.2f}".format(reward))

    def close(self):
        '''
        Implement the close function to clean up (if needed)
        '''
        pass


if __name__ == "__main__":
    # Create the serverless environment
    rlss_env = ServerlessEnv()
    
    # Reset the environment to the initial state
    observation, info = rlss_env.reset()
    
    os.system('clear')
    for i, (row1, row2) in enumerate(zip(rlss_env._container_matrix, rlss_env._exectime_matrix), start=1):
        print(f"Service {i}: {row1[0]} containers\t- {row2[0]}s to execute each request")
    
    # Perform random actions
    i = 0
    while (True):
        i += 1
        print("----------------------------------------")
        time.sleep(10)  # Gives action every 10 seconds
        action = rlss_env.action_space.sample()  # Random action
        print(action[1][0])
        # print("Action:\n", action)
        observation, reward, terminated, truncated, info = rlss_env.step(action)
        print(f"Round: {i}, Done: {terminated}")
        rlss_env.render()
        if (terminated): 
            print("----------------------------------------")
            break
        else: continue
    
    os._exit(1)