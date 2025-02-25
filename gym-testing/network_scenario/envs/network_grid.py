import numpy as np

import gymnasium as gym
from gymnasium import spaces


class NetworkGridEnv(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None, size=5):
        super(NetworkGridEnv, self).__init__()
        '''
        Define environment parameters
        '''
        self.size = size  # The size of the square grid
        self.num_bs = self.size ** 2
        self.num_actions = self.num_bs * 17

        '''
        - Observations (state space) are dictionaries with 2 traffic matrices "traffic_demand" and "traffic_state".
        - There are 17 actions for each BS, load can be split equally in up to 4 directions.
        '''
        self.observation_space = spaces.Dict(
            {
                "traffic_demand": spaces.Box(low=0, high=1, shape=(self.size, self.size), dtype=np.float32),
                "traffic_state": spaces.Box(low=0, high=1, shape=(self.size, self.size), dtype=np.float32)
            }
        )
        
        self.action_space = spaces.Discrete(self.num_actions)
        
        '''
        Initialize the state and other variables
        '''
        self.current_time = 0  # Start at time 0
        self.max_steps = 24  # 24 steps for 24 hours in a day
        self.active_bs = np.ones((self.size, self.size), dtype=bool)
        self.prev_bs = (-1, -1)  # Set an initial value
        self._demand_matrix = np.full((self.size, self.size), 0.1)
        self._state_matrix = np.full((self.size, self.size), 0.1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_reward(self):
        # Calculate traffic loss
        loss = np.sum(self._demand_matrix) - np.sum(self._state_matrix)
        
        # Calculate energy consumption
        energy_csm = []
        for i in range(self.size):
            for j in range(self.size):
                if (self._state_matrix[i, j] > 1):
                    self._state_matrix[i, j] = 1
                    energy_csm.append(130 + 4.7 * 20)
                elif (0 < self._state_matrix[i, j] <= 1):
                    energy_csm.append(130 + 4.7 * 20 * self._state_matrix[i, j])
                else:
                    energy_csm.append(75)
        
        max_energy_csm = np.max(energy_csm)
        reward = - 100 * loss
        for i in range(self.num_bs):
            reward += (max_energy_csm - energy_csm[i])
        
        return reward
        
    def _get_obs(self):
        return {"traffic_demand": self._demand_matrix, "traffic_state": self._state_matrix}
    
    def _get_info(self):
        energy_csm = []
        all_on_energy_csm = []
        for i in range(self.size):
            for j in range(self.size):
                if (self._state_matrix[i, j] > 1):
                    energy_csm.append(130 + 4.7 * 20)
                    all_on_energy_csm.append(130 + 4.7 * 20)
                elif (0 < self._state_matrix[i, j] <= 1):
                    energy_csm.append(130 + 4.7 * 20 * self._state_matrix[i, j])
                    all_on_energy_csm.append(130 + 4.7 * 20 * self._state_matrix[i, j])
                else:
                    energy_csm.append(75)
        return {
            "traffic_coverage": np.sum(self._state_matrix) / np.sum(self._demand_matrix) * 100,
            "energy_saving": (np.sum(all_on_energy_csm) - np.sum(energy_csm)) / np.sum(all_on_energy_csm) * 100
        }
    
    def reset(self, seed=None, options=None):
        '''
        Initialize the environment
        '''
        self.current_time = 0
        self.active_bs = np.ones((self.size, self.size), dtype=bool)
        self.prev_bs = -1, -1  # Set an initial value
        self._demand_matrix = np.full((self.size, self.size), 0.1)
        self._state_matrix = np.full((self.size, self.size), 0.1)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        '''
        Check if action is within the valid range [0, 424]
        '''
        if action < 0 or action > 424:
            raise ValueError("Invalid action. Action must be in the range [0, 424].")
        
        '''
        Determine the action and the BS location to perform the action
        '''
        bs_row, bs_col, bs_action = action // 17, action % 17, action % 17
        bs_row, bs_col = bs_row // self.size, bs_row % self.size
        
        '''
        Implement action masking to ensure valid actions and update traffic state based on load shifts
        '''
        if ((bs_row, bs_col) == self.prev_bs):
        # Check if the current BS is the same as the previous BS
            terminated = self.current_time >= self.max_steps
            truncated = False
            reward = self._get_reward()
            observation = self._get_obs()
            info = self._get_info()  
            return observation, reward, terminated, truncated, info
        
        if (bs_action == 0):
            # Action 0: BS turning ON
            self.active_bs[bs_row, bs_col] = 1
        elif (bs_action >= 1 and bs_action <= 15):
            # Action 1-15: Bs turning OFF and shifting loads
            if (self.active_bs[bs_row, bs_col] == 0):
                # Do not turn off an already deactivated BS
                terminated = self.current_time >= self.max_steps
                truncated = False
                reward = self._get_reward()
                observation = self._get_obs()
                info = self._get_info()  
                return observation, reward, terminated, truncated, info
            else:
                self.active_bs[bs_row, bs_col] = 0
                self._state_matrix[bs_row, bs_col] = 0
                bs_left, bs_top = (bs_action & 0b1000) >> 3, (bs_action & 0b0100) >> 2
                bs_right, bs_bottom = (bs_action & 0b0010) >> 1, (bs_action & 0b0001)
                corner_idx = self.size - 1
                
                # BSs at edge may have less than 4 neighbors, don't share load to OFF BS
                if ((bs_row, bs_col) == (0, 0)):
                    shift_load = bs_right * self.active_bs[0, 1] + bs_bottom * self.active_bs[1, 0]
                    if (shift_load != 0):
                        if (self.active_bs[0, 1] == 0): self._state_matrix[1, 0] += self._state_matrix[bs_row, bs_col]
                        elif (self.active_bs[1, 0] == 0): self._state_matrix[0, 1] += self._state_matrix[bs_row, bs_col]
                        else:
                            self._state_matrix[0, 1] += (self._state_matrix[bs_row, bs_col] / shift_load)
                            self._state_matrix[1, 0] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    
                elif ((bs_row, bs_col) == (0, corner_idx)):
                    shift_load = bs_left * self.active_bs[0, corner_idx - 1] + bs_bottom * self.active_bs[1, corner_idx]
                    if (shift_load != 0):
                        if (self.active_bs[0, corner_idx - 1] == 0): self._state_matrix[1, corner_idx] += self._state_matrix[bs_row, bs_col]
                        elif (self.active_bs[1, corner_idx] == 0): self._state_matrix[0, corner_idx - 1] += self._state_matrix[bs_row, bs_col]
                        else:
                            self._state_matrix[0, corner_idx - 1] += (self._state_matrix[bs_row, bs_col] / shift_load)
                            self._state_matrix[1, corner_idx] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    
                elif ((bs_row, bs_col) == (corner_idx, 0)):
                    shift_load = bs_top * self.active_bs[corner_idx - 1, 0] + bs_right * self.active_bs[corner_idx, 1]
                    if (shift_load != 0):
                        if (self.active_bs[corner_idx - 1, 0] == 0): self._state_matrix[corner_idx, 1] += self._state_matrix[bs_row, bs_col]
                        elif (self.active_bs[corner_idx, 1] == 0): self._state_matrix[corner_idx - 1, 0] += self._state_matrix[bs_row, bs_col]
                        else:
                            self._state_matrix[corner_idx - 1, 0] += (self._state_matrix[bs_row, bs_col] / shift_load)
                            self._state_matrix[corner_idx, 1] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    
                elif ((bs_row, bs_col) == (corner_idx, corner_idx)):
                    shift_load = bs_top * self.active_bs[corner_idx - 1, corner_idx] + bs_left * self.active_bs[corner_idx, corner_idx - 1]
                    if (shift_load != 0):
                        if (self.active_bs[corner_idx - 1, corner_idx] == 0): self._state_matrix[corner_idx, corner_idx - 1] += self._state_matrix[bs_row, bs_col]
                        elif (self.active_bs[corner_idx, corner_idx - 1] == 0): self._state_matrix[corner_idx - 1, corner_idx] += self._state_matrix[bs_row, bs_col]
                        else:
                            self._state_matrix[corner_idx - 1, corner_idx] += (self._state_matrix[bs_row, bs_col] / shift_load)
                            self._state_matrix[corner_idx, corner_idx - 1] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    
                elif (bs_row == 0):
                    shift_load = bs_left * self.active_bs[0, bs_col - 1] + bs_right * self.active_bs[0, bs_col + 1] + bs_bottom * self.active_bs[1, bs_col]
                    if (shift_load != 0):
                        for dr, dc in [(0, -1), (0, 1), (1, 0)]:
                            if (self.active_bs[dr, bs_col + dc] != 0):
                                self._state_matrix[dr, bs_col + dc] += (self._state_matrix[bs_row, bs_col] / shift_load)

                elif (bs_row == corner_idx):
                    shift_load = bs_left * self.active_bs[corner_idx, bs_col - 1] + bs_right * self.active_bs[corner_idx, bs_col + 1] + bs_top * self.active_bs[corner_idx - 1, bs_col]
                    if (shift_load != 0):          
                        for dr, dc in [(0, -1), (0, 1), (-1, 0)]:
                            if (self.active_bs[corner_idx + dr, bs_col + dc] != 0):
                                self._state_matrix[corner_idx + dr, bs_col + dc] += (self._state_matrix[bs_row, bs_col] / shift_load)

                elif (bs_col == 0):
                    shift_load = bs_right * self.active_bs[bs_row, 1] + bs_top * self.active_bs[bs_row - 1, 0] + bs_bottom * self.active_bs[bs_row + 1, 0]
                    if (shift_load != 0):       
                        for dr, dc in [(-1, 0), (1, 0), (0, 1)]:
                            if (self.active_bs[bs_row + dr, dc] != 0):
                                self._state_matrix[bs_row + dr, dc] += (self._state_matrix[bs_row, bs_col] / shift_load)

                elif (bs_col == corner_idx):
                    shift_load = bs_left * self.active_bs[bs_row, corner_idx - 1] + bs_top * self.active_bs[bs_row - 1, corner_idx] + bs_bottom * self.active_bs[bs_row + 1, corner_idx]
                    if (shift_load != 0): 
                        for dr, dc in [(-1, 0), (1, 0), (0, -1)]:
                            if (self.active_bs[bs_row + dr, corner_idx + dc] != 0):
                                self._state_matrix[bs_row + dr, corner_idx + dc] += (self._state_matrix[bs_row, bs_col] / shift_load)

                else:
                    shift_load = bs_left * self.active_bs[bs_row, bs_col - 1] + bs_top * self.active_bs[bs_row - 1, bs_col] + bs_right * self.active_bs[bs_row, bs_col + 1] + bs_bottom * self.active_bs[bs_row + 1, bs_col]
                    if (shift_load != 0): 
                        for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                            if (self.active_bs[bs_row + dr, bs_col + dc] != 0):
                                self._state_matrix[bs_row + dr, bs_col + dc] += (self._state_matrix[bs_row, bs_col] / shift_load)
               
        else: 
            pass  # Action 16: Do nothing
        
        '''
        Increment the time step 
        '''
        self.current_time += 1        
        
        '''
        An episode is done after 24 steps
        '''
        terminated = self.current_time >= self.max_steps
        truncated = False
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()
        '''
        Update the previous BS
        '''
        self.prev_bs = bs_row, bs_col
        
        return observation, reward, terminated, truncated, info
        
    
    def render(self):
        '''
        Implement a visualization method (if needed)
        '''
        info = self._get_info()
        print("Performance Metrics:")
        print("- Traffic Coverage: {:.2f}%".format(info["traffic_coverage"]))
        print("- Energy Saving: {:.2f}%".format(info["energy_saving"]))
    
    def close(self):
        '''
        Implement the close function to clean up (if needed)
        '''
        pass


if __name__ == "__main__":
    # Create the network grid environment
    network_env = NetworkGridEnv()
    
    # Reset the environment to the initial state
    observation, info = network_env.reset()
    
    # Perform some random actions to see the state of BSs and their loads
    while (True):
        print("----------------------------------------")
        action = network_env.action_space.sample()  # Random action
        print("Action:", action)
        observation, reward, terminated, truncated, info = network_env.step(action)
        print(f"Reward: {reward}, Done: {terminated}")
        network_env.render()
        if terminated: 
            print("----------------------------------------")
            break
        else: continue