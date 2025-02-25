# import enum
# import numpy as np

# class Actions(enum.Enum):
#     previous_state = -1
#     current_state = 0
#     next_state = 1

# action = len(Actions)

# n = 5
# result_matrix = np.hstack((
#     np.random.randint(0, 256, size=(n, 1)),
#     np.zeros((n, n-1), dtype=np.int16)
# ))

# def check():
#     if (n > 6):
#         pass
#     else:
#         print("n <= 6")

#     print("Initialized Matrix:")
#     print(result_matrix)

# check()

# import gym
# from gym import spaces
# import numpy as np

# class CustomEnvironment(gym.Env):
#     def __init__(self, size=3):
#         self.size = size
#         self.max_container = 256
#         self.num_states = 5
#         self.num_resources = 3

#         '''
#         Define action space containing two matrices by combining them into a Tuple space
#         '''
#         self._action_coefficient = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int16)
#         self._action_unit = spaces.Box(low=-1, high=1, shape=(self.size, self.num_states), dtype=np.int16)
#         self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))

#         # Set the main diagonal elements of _action_coefficient to be in the range [0, self.max_container]
#         np.fill_diagonal(self._action_coefficient.low, 0)
#         np.fill_diagonal(self._action_coefficient.high, self.max_container)

#         # Set the last column of the _action_unit to be always zero
#         self._action_unit.low[:, -1] = 0
#         self._action_unit.high[:, -1] = 0

#         self._generate_valid_custom_matrix()  # Generate a valid custom matrix

#         self._resource_matrix = np.zeros((self.num_resources, 1), dtype=np.int16)
#         self._resource_matrix[0, 0] = 100


#     def _generate_valid_custom_matrix(self):
#         while True:
#             random_matrix = np.random.randint(-1, 2, size=(self.size, self.num_states-1))  # Generate a random matrix
#             row_sums = np.sum(random_matrix, axis=1)  # Ensure the row sum is 0
#             if np.all(row_sums == 0):  # If row sum is 0, assign the matrix to _action_unit
#                 self._action_unit.low[:, :-1] = random_matrix
#                 self._action_unit.high[:, :-1] = random_matrix
#                 break

#     # ... (rest of the class methods remain the same)

# # Example usage:
# rlss_env = CustomEnvironment(size=3)  # Create an instance of the custom environment with size = 3

# # Generate a sample action using action_space.sample()
# action = rlss_env.action_space.sample()

# # Display the sampled action
# print("Action Coefficient Matrix:")
# print(action[0])
# print("Action Units Matrix:")
# print(action[1])
# print("Sampled Action:")
# print(action[0] @ action[1])

# print(rlss_env._resource_matrix)

# # Create a sample matrix (3x3) for illustration purposes
# matrix = np.array([[1, 2, 3],
#                    [4, 0, 6],
#                    [7, 8, 0]])

# # Check if any element is less than 0
# any_less_than_zero = np.any(matrix < 0)

# print("Any element is less than 0:", any_less_than_zero)

# sub_matrix = np.array([1, 5, 2])
# matrix *= sub_matrix

# print(matrix)

# print(matrix[0, 0])

# a, b = 100, 200

# print(a)
# print(b)

# for i, row in enumerate(matrix):
#     print(f"{i + 1}: {row[0]}")


# # Example matrices
# matrix1 = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])

# matrix2 = np.array([
#     [11, 12, 13],
#     [14, 15, 16],
#     [17, 18, 19]
# ])

# for i, (row1, row2) in enumerate(zip(matrix1, matrix2), start=1):
#     for j, (elem1, elem2) in enumerate(zip(row1, row2), start=1):
#         print(f"Matrix 1 - Row: {i}, Col: {j}, Value: {elem1}")
#         print(f"Matrix 2 - Row: {i}, Col: {j}, Value: {elem2}")

# import numpy as np

# array_set = {
#     'array_1': np.array([-1, 1, 0, 0, 0]),
#     'array_2': np.array([1, -1, 0, 0, 0]),
#     'array_3': np.array([0, -1, 1, 0, 0]),
#     'array_4': np.array([0, 1, -1, 0, 0]),
#     'array_5': np.array([0, 0, -1, 1, 0]),
#     'array_6': np.array([0, 0, 1, -1, 0]),
# }

# def initialize_matrix(size):
#     keys = list(array_set.keys())
#     matrix = np.array([array_set[np.random.choice(keys)] for _ in range(size)])
#     return matrix

# # Example usage:
# result_matrix = initialize_matrix(4)
# print(result_matrix)
# print(result_matrix[0])

# print(np.array_equal(result_matrix[0], array_set["array_1"]))

# import numpy as np

# import gymnasium as gym
# from gymnasium import spaces


# class Transitions:
#     trans_0 = np.array([-1, 1, 0, 0, 0])  # N -> L0
#     trans_1 = np.array([1, -1, 0, 0, 0])  # L0 -> N
#     trans_2 = np.array([0, -1, 1, 0, 0])  # L0 -> L1
#     trans_3 = np.array([0, 1, -1, 0, 0])  # L1 -> L0
#     trans_4 = np.array([0, 0, -1, 1, 0])  # L1 -> L2
#     trans_5 = np.array([0, 0, 1, -1, 0])  # L2 -> L1
#     trans_6 = np.array([1, -1, -1, 1, 0])  # L0 -> N and L1 -> L2
#     trans_7 = np.array([-1, 1, 1, -1, 0])  # N -> L0 and L2 -> L1
#     trans_8 = np.array([1, -1, 1, -1, 0])  # L0 -> N and L2 -> L1
#     trans_9 = np.array([-1, 1, -1, 1, 0])


# class ServerlessEnv(gym.Env):
#     metadata = {}

#     def __init__(self, render_mode=None, size=4):
#         super(ServerlessEnv, self).__init__()
#         """
#         Define environment parameters
#         """
#         self.size = size  # The number of services
#         self.num_states = (
#             5  # The number of states in a container's lifecycle (N, L0, L1, L2, A)
#         )
#         self.num_resources = 3  # The number of resource parameters (RAM, GPU, CPU)
#         self.max_container = 256

#         self.timeout = 10  # Set timeout value = 10s
#         self.container_lifetime = 43200  # Set lifetime of a container = 1/2 day
#         self.limited_ram = 64  # Set limited amount of RAM (server) = 64GB
#         self.limited_request = 128

#         """
#         Define observations (state space)
#         """
#         self.observation_space = spaces.Dict(
#             {
#                 "execution_times": spaces.Box(
#                     low=1, high=10, shape=(self.size, 1), dtype=np.int16
#                 ),
#                 "request_quantity": spaces.Box(
#                     low=0,
#                     high=self.limited_request,
#                     shape=(self.size, 1),
#                     dtype=np.int16,
#                 ),
#                 "remained_resource": spaces.Box(
#                     low=0,
#                     high=self.limited_ram,
#                     shape=(self.num_resources, 1),
#                     dtype=np.int16,
#                 ),
#                 "container_traffic": spaces.Box(
#                     low=0,
#                     high=self.max_container,
#                     shape=(self.size, self.num_states),
#                     dtype=np.int16,
#                 ),
#             }
#         )

#         """
#         Define action space containing two matrices by combining them into a Tuple space
#         """
#         self._action_coefficient = spaces.Box(
#             low=0, high=0, shape=(self.size, self.size), dtype=np.int16
#         )
#         self._action_unit = spaces.Box(
#             low=-1, high=1, shape=(self.size, self.num_states), dtype=np.int16
#         )
#         self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))

#         # Set the main diagonal elements of _action_coefficient to be in the range [0, self.max_container]
#         np.fill_diagonal(self._action_coefficient.low, 0)
#         np.fill_diagonal(self._action_coefficient.high, self.max_container)

#         # Set the last column of the _action_unit to be always zero and the sum of the elements in a row of the _action_unit = 0 using _get_units()
#         # self._action_unit.low[:, -1] = 0
#         # self._action_unit.high[:, -1] = 0
#         self._get_units()

#         """
#         Initialize the state and other variables
#         """
#         self.current_time = 0  # Start at time 0
#         self._custom_request = np.random.randint(
#             0, 64, size=(self.size, 1)
#         )  # Randomly set the number of incoming requests every Δt seconds
#         self._pending_request = np.zeros(
#             (self.size, 1), dtype=np.int16
#         )  # Set an initial value
#         self._ram_required_matrix = np.array(
#             [0, 0, 0, 0.9, 2]
#         )  # Set the required RAM each state
#         self._action_matrix = np.zeros(
#             (self.size, self.num_states), dtype=np.int16
#         )  # Set an initial value
#         self._exectime_matrix = np.random.randint(2, 16, size=(self.size, 1))
#         self._request_matrix = np.zeros((self.size, 1), dtype=np.int16)
#         self._resource_matrix = np.ones((self.num_resources, 1), dtype=np.int16)
#         self._resource_matrix[0, 0] = self.limited_ram
#         self._container_matrix = np.hstack(
#             (
#                 np.random.randint(
#                     0, self.max_container, size=(self.size, 1)
#                 ),  # Initially the containers are in Null state
#                 np.zeros((self.size, self.num_states - 1), dtype=np.int16),
#             )
#         )

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode

#     def _get_units(self):
#         """
#         Define a function to create a random matrix such that the sum of the elements in a row = 0
#         """
#         # while True:
#         #     random_matrix = np.random.randint(-1, 2, size=(self.size, self.num_states-1))  # Generate a random matrix
#         #     row_sums = np.sum(random_matrix, axis=1)  # Ensure the row sum is 0
#         #     if np.all(row_sums == 0):  # If row sum is 0, assign the matrix to _action_unit
#         #         self._action_unit.low[:, :-1] = random_matrix
#         #         self._action_unit.high[:, :-1] = random_matrix
#         #         break
#         array_set = [
#             getattr(Transitions, attr)
#             for attr in dir(Transitions)
#             if not attr.startswith("__")
#         ]
#         self._action_unit = np.array(
#             [array_set[np.random.randint(0, len(array_set))] for _ in range(self.size)]
#         )


# # Example usage:
# env = ServerlessEnv()
# result_matrix = env._action_unit
# print(result_matrix)

# # Example 4x1 matrix
# matrix_4x1 = np.array([[5], [2], [7], [3]])  # Replace this with your matrix

# # Get elements and sort them
# sorted_elements = np.sort(matrix_4x1, axis=0)  # Sort the elements along the columns

# # Create a new 4x1 matrix with the sorted elements
# new_sorted_matrix = sorted_elements

# print("\nNew 4x1 matrix with sorted elements:")
# print(new_sorted_matrix - 1)

# for i in range(4):
#     print(new_sorted_matrix[i])

# import numpy as np
# from gym import spaces


# class Transitions:
#     trans_0 = np.array([-1, 1, 0, 0, 0])  # N -> L0
#     trans_1 = np.array([1, -1, 0, 0, 0])  # L0 -> N
#     trans_2 = np.array([0, -1, 1, 0, 0])  # L0 -> L1
#     trans_3 = np.array([0, 1, -1, 0, 0])  # L1 -> L0
#     trans_4 = np.array([0, 0, -1, 1, 0])  # L1 -> L2
#     trans_5 = np.array([0, 0, 1, -1, 0])  # L2 -> L1
#     trans_6 = np.array([1, -1, -1, 1, 0])  # L0 -> N and L1 -> L2
#     trans_7 = np.array([-1, 1, 1, -1, 0])  # N -> L0 and L2 -> L1
#     trans_8 = np.array([1, -1, 1, -1, 0])  # L0 -> N and L2 -> L1
#     trans_9 = np.array([-1, 1, -1, 1, 0])  # N -> L0 and L1 -> L2


# class CustomEnvironment:
#     def __init__(self, size, num_states):
#         self.size = size
#         self.num_states = num_states
#         self._action_coefficient = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int16)
#         self._action_unit = spaces.Box(low=-1, high=1, shape=(self.size, self.num_states), dtype=np.int16)
#         self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))

#     def _get_units(self):
#         array_set = [getattr(Transitions, attr) for attr in dir(Transitions) if not attr.startswith("__")]
#         random_matrix = np.array([array_set[np.random.randint(0, len(array_set))] for _ in range(self.size)])
#         self._action_unit = spaces.Box(low=np.min(random_matrix), high=np.max(random_matrix), shape=random_matrix.shape, dtype=np.int16)

# # Example usage:
# env = CustomEnvironment(size=4, num_states=5)
# env._get_units()
# action = env.action_space
# print("Action Space:", action[0])

# income_request_matrix = np.full((10, 1), 500, dtype=np.int32)

# print("Income Request Matrix:")
# print(income_request_matrix[0])

# from scipy.stats import poisson ## to calculate the passion distribution
# import numpy as np ## to prepare the data
# import pandas as pd ## to prepare the data
# import matplotlib.pyplot as plt ## to create plots

# #generate Poisson distribution with sample size 10000

# d_rvs = pd.Series(np.random.poisson(640, size=24))
# d_rvs = pd.Series(poisson.rvs(640, size=24))

# # print(d_rvs.mean())
# # print(d_rvs)

# data = d_rvs.to_dict()

# print(data)
# fig, ax = plt.subplots(figsize=(16, 6))
# ax.bar(range(len(data)), list(data.values()), align='center')
# plt.xticks(range(len(data)), list(data.keys()))
# plt.show()

# import random
# import math
# import statistics
# import matplotlib.pyplot as plt

# _lambda = 5
# _num_events = 100
# _event_num = []
# _inter_event_times = []
# _event_times = []
# _event_time = 0

# print('EVENT_NUM,INTER_EVENT_T,EVENT_T')

# for i in range(_num_events):
# 	_event_num.append(i)
# 	#Get a random probability value from the uniform distribution's PDF
# 	n = random.random()

# 	#Generate the inter-event time from the exponential distribution's CDF using the Inverse-CDF technique
# 	_inter_event_time = -math.log(1.0 - n) / _lambda
# 	_inter_event_times.append(_inter_event_time)

# 	#Add the inter-event time to the running sum to get the next absolute event time
# 	_event_time = _event_time + _inter_event_time
# 	_event_times.append(_event_time)

# 	#print it all out
# 	print(str(i) +',' + str(_inter_event_time) + ',' + str(_event_time))

# #plot the inter-event times
# fig = plt.figure()
# fig.suptitle('Times between consecutive events in a simulated Poisson process')
# plot, = plt.plot(_event_num, _inter_event_times, 'bo-', label='Inter-event time')
# plt.legend(handles=[plot])
# plt.xlabel('Index of event')
# plt.ylabel('Time')
# plt.show()


# #plot the absolute event times
# fig = plt.figure()
# fig.suptitle('Absolute times of consecutive events in a simulated Poisson process')
# plot, = plt.plot(_event_num, _event_times, 'bo-', label='Absolute time of event')
# plt.legend(handles=[plot])
# plt.xlabel('Index of event')
# plt.ylabel('Time')
# plt.show()

# _interval_nums = []
# _num_events_in_interval = []
# _interval_num = 1
# _num_events = 0

# print('INTERVAL_NUM,NUM_EVENTS')

# for i in range(len(_event_times)):
# 	_event_time = _event_times[i]
# 	if _event_time <= _interval_num:
# 		_num_events += 1
# 	else:
# 		_interval_nums.append(_interval_num)
# 		_num_events_in_interval.append(_num_events)

# 		print(str(_interval_num) +',' + str(_num_events))

# 		_interval_num += 1

# 		_num_events = 1

# #print the mean number of events per unit time
# print(statistics.mean(_num_events_in_interval))

# #plot the number of events in consecutive intervals
# fig = plt.figure()
# fig.suptitle('Number of events occurring in consecutive intervals in a simulated Poisson process')
# plt.bar(_interval_nums, _num_events_in_interval)
# plt.xlabel('Index of interval')
# plt.ylabel('Number of events')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# '''
# Hàm generate_poisson_events() mô phỏng quy trình Poisson bằng cách 
# tạo các sự kiện với tốc độ trung bình (rate) nhất định trong một khoảng thời gian xác định (time_duration).
# '''
# def generate_poisson_events(rate, time_duration):
#     # Tính tổng số sự kiện bằng phân phối Poisson
#     num_events = np.random.poisson(rate * time_duration)
    
#     # Tạo ra thời gian đến giữa các sự kiện bằng phân phối mũ với giá trị trung bình là 1.0 / rate
#     inter_arrival_times = np.random.exponential(1.0 / rate, num_events)
    
#     # Cộng dồn thời gian giữa các lần đến để có được thời gian đến của sự kiện
#     event_times = np.cumsum(inter_arrival_times)
    
#     # Trả về số lượng sự kiện, thời gian sự kiện và thời gian giữa các lần đến tương ứng
#     print(num_events, event_times, inter_arrival_times)
#     return num_events, event_times, inter_arrival_times

# def plot_non_sequential_poisson(num_events, event_times, inter_arrival_times, rate, time_duration):
#     fig, axs = plt.subplots(1, 2, figsize=(15, 6))
#     fig.suptitle(f'Poisson Process Simulation (λ = {rate}, Duration = {time_duration} seconds)\n', fontsize=16)

#     axs[0].step(event_times, np.arange(1, num_events + 1), where='post', color='blue')
#     axs[0].set_xlabel('Time')
#     axs[0].set_ylabel('Event Number')
#     axs[0].set_title(f'Poisson Process Event Times\nTotal: {num_events} events\n')
#     axs[0].grid(True)

#     axs[1].hist(inter_arrival_times, bins=20, color='green', alpha=0.5)
#     axs[1].set_xlabel('Inter-Arrival Time')
#     axs[1].set_ylabel('Frequency')
#     axs[1].set_title(f'Histogram of Inter-Arrival Times\nMEAN: {np.mean(inter_arrival_times):.2f} | STD: {np.std(inter_arrival_times):.2f}\n')
#     axs[1].grid(True, alpha=0.5)
    
#     plt.tight_layout()
#     plt.show()

# def plot_sequential_poisson(num_events_list, event_times_list, inter_arrival_times_list, rate, time_duration):
#     fig, axs = plt.subplots(1, 2, figsize=(15, 6))
#     fig.suptitle(f'Poisson Process Simulation (Duration = {time_duration} seconds)\n', fontsize=16)

#     axs[0].set_xlabel('Time')
#     axs[0].set_ylabel('Event Number')
#     axs[0].set_title(f'Poisson Process Event Times')
#     axs[0].grid(True)

#     axs[1].set_xlabel('Inter-Arrival Time')
#     axs[1].set_ylabel('Frequency')
#     axs[1].set_title(f'Histogram of Inter-Arrival Times')
#     axs[1].grid(True, alpha=0.5)

#     color_palette = plt.get_cmap('tab20')
#     colors = [color_palette(i) for i in range(len(rate))]

#     for n, individual_rate in enumerate(rate):
#         num_events = num_events_list[n]
#         event_times = event_times_list[n]
#         inter_arrival_times = inter_arrival_times_list[n]

#         axs[0].step(event_times, np.arange(1, num_events + 1), where='post', color=colors[n], label=f'λ = {individual_rate}, Total Events: {num_events}')
#         axs[1].hist(inter_arrival_times, bins=20, color=colors[n], alpha=0.5, label=f'λ = {individual_rate}, MEAN: {np.mean(inter_arrival_times):.2f}, STD: {np.std(inter_arrival_times):.2f}')

#     axs[0].legend()
#     axs[1].legend()

#     plt.tight_layout()
#     plt.show()

# def poisson_simulation(rate, time_duration, show_visualization=True):
#     if isinstance(rate, int):
#         num_events, event_times, inter_arrival_times = generate_poisson_events(rate, time_duration)
        
#         if show_visualization:
#             plot_non_sequential_poisson(num_events, event_times, inter_arrival_times, rate, time_duration)
#         else:
#             return num_events, event_times, inter_arrival_times

#     elif isinstance(rate, list):
#         num_events_list = []
#         event_times_list = []
#         inter_arrival_times_list = []

#         for individual_rate in rate:
#             num_events, event_times, inter_arrival_times = generate_poisson_events(individual_rate, time_duration)
#             num_events_list.append(num_events)
#             event_times_list.append(event_times)
#             inter_arrival_times_list.append(inter_arrival_times)

#         if show_visualization:
#             plot_sequential_poisson(num_events_list, event_times_list, inter_arrival_times_list, rate, time_duration)
#         else:
#             return num_events_list, event_times_list, inter_arrival_times_list

# # Example usage
# # poisson_simulation(rate=1, time_duration=3600) # For single lambda rate (non-sequential)
# # poisson_simulation(rate=[2, 4, 6, 10], time_duration=10) # For multiple lambda rate (sequential)

# import numpy as np
# import time

# def add_to_B_after_delay(A, B):
#     # Flatten A to find the unique elements and sort them
#     unique_elements = np.unique(A)
#     sorted_elements = np.sort(unique_elements)
    
#     # Iterate through each element in sorted order
#     for i in range (0, len(sorted_elements)):
#         # Find the indices where the element occurs in A
#         indices = np.where(A == sorted_elements[i])
        
#         # Wait for the delay before adding to B
#         delay = sorted_elements[i] - sorted_elements[i-1] if i != 0 else 0
#         time.sleep(delay)
        
#         # Add 1 unit to the corresponding rows of B
#         for idx in indices[0]:
#             B[idx] += 1
        
#         print(f"After {delay} seconds: B =\n{B}\n")

# # Define matrices A and B
# A = np.array([[1, 2, 3, 4, 5, 6], [1.1, 1.8, 2.9, 4.1, 5.2, 5.9]])
# B = np.array([[0], [0]])

# # Perform the task
# add_to_B_after_delay(A, B)

# # Example array
# arr = np.array([1, 0, 2, 0, 3, 0, 4, 0, 5])

# # Delete zero elements
# non_zero_elements = arr[arr != 0]

# print("Array after deleting zero elements:", non_zero_elements)

# # Import data to a matrix
# n = 4
# rate = 1
# time_duration = 10
# incoming_matrix = np.zeros((n, 10))

# for i in range(n):
#     num_events, event_times, inter_arrival_times = generate_poisson_events(rate=rate, time_duration=time_duration)
#     if (num_events > 10):
#         temp_matrix = np.zeros((n, num_events))
#         row_index = 0
#         col_index = 0
#         temp_matrix[row_index:row_index + incoming_matrix.shape[0], col_index:col_index + incoming_matrix.shape[1]] = incoming_matrix
#         incoming_matrix = temp_matrix
   
#     incoming_matrix[i, :num_events] = inter_arrival_times

# print("Incoming requests:")    
# print(incoming_matrix)

# num_events, event_times, inter_arrival_times = generate_poisson_events(rate=640/3600, time_duration=86400)

