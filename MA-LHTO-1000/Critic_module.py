import math

import numpy as np

import gv_c as gvc
import gloable_variation as gv
from config import config
import critic_End_User as ceu
#import critic_MES as cm
N =config.get('Dev_dev')
M =config.get('Dev_edge')
Task_n = config.get('Task')
TimeZone = gv.TimeZone
def critic_en(time,x,y,f,velocity,K):
    value,complete=[],[]
    for ind_x in range(K):
        complete_time, fragment = ceu.critic(time, x, y, f, velocity, ind_x)
        wait_size_ES = gvc.wait_size_ES
        stable = 0
        for dev in range(M):
            for type in range(len(wait_size_ES[dev])):
                stable += Task_n * N * wait_size_ES[dev][type][time + 1] * (10 ** 5) / ((393216 * 3) * (10 ** 9))
        value_index = 10 * (0.7 * 1 / (1 + math.exp(-complete_time)) - 0.3 * fragment) - 1 / (1 + math.exp(-stable))
        value.append([ind_x, value_index])
        '''
                    info = gvc.info
                    gen_task,ability = TimeZone[time],[]
                    for task in gen_task:
                        result = info.loc[(info.index == task)]
                        delay = result['end'].values[0] - result['start'].values[0]
                        ma = math.exp(-result['memory_mib'].values[0]/delay)
                        ability.append(1/(1+ma))
                    value.append([ind_x,ind_y,0.7*sum(ability)+0.3*fragment])
                    '''
            
    max_index,value_max=None,None
    for index in range(len(value)):
        current_value = value[index][1]
        if value_max is None or current_value > value_max:
            value_max = current_value
            max_index = value[index]
    return max_index[0],max_index[1]