from enum import Enum
import numpy as np

class Actions(Enum):
   pass

for i in range(25):
    for j in range(17):
        if (j == 0):
            setattr(Actions, f"ACTION_{i+1}_{j}", "Turning ON")
        elif (j == 16):
            setattr(Actions, f"ACTION_{i+1}_{j}", "Do nothing")
        else:
            setattr(Actions, f"ACTION_{i+1}_{j}", "Turning OFF and shift load")
        
action = Actions.ACTION_1_0

action = 118
bs_actions = [action % 17 if i == (action // 17) else 16 for i in range(25)]

prev_bs = 1, 2
bs_row, bs_col = 2, 2

action = action % 17
bs_left   = (action & 0b1000) >> 3
bs_top    = (action & 0b0100) >> 2
bs_right  = (action & 0b0010) >> 1
bs_bottom = (action & 0b0001)

arr = []

for i in range(25):
    arr.append(i)
    
print(np.sum(arr))