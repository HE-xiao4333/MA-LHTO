import copy
import numpy as np
import importlib
import math
from config import config
import gv_c as gvc
import critic_MES as MES

info = gvc.info
task_queue = gvc.task_queue
wait_tackel_queue = gvc.wait_tackel_queue
wait_offload_queue = gvc.wait_offload_queue
wait_size_ES = gvc.wait_size_ES
f_local = gvc.f_local
F_es =gvc.F_es
def global_update():
    global info,task_queue,sub_task_queue,wait_tackel_queue,wait_offload_queue,wait_size_ES,wait_local_size
    importlib.reload(gvc)
    info = gvc.info
    task_queue = gvc.task_queue
    wait_tackel_queue = gvc.wait_tackel_queue
    wait_offload_queue = gvc.wait_offload_queue
    wait_size_ES = gvc.wait_size_ES
    wait_local_size = gvc.wait_local_size



N=config.get('Dev_dev')
M = config.get('Dev_edge')
Band = config.get("B")
f_local_max=config.get('f_local_max')
f_local_min = config.get('f_local_min')
P_tran_min = config.get('P_tran_min')
P_tran_max = config.get('P_tran_max')
alpha_1=config.get('alpha_1')
coverage = config.get('Scope')
edge_position = config.get("Edge_position")
speed = config.get("Speed")
noisy = config.get("noisy")
loss_exponent = 3
light = 3 * 10 ** 8  # 光速
Ad = 3  # 接收天线增益？
fc = 915 * 10 ** 6  # 无线电波的载频。？
K= 5



def updage_offload_decision(time,x,y,dev):
    wait_task = task_queue[dev][time]
    if not wait_task:
        return
    for z in range(len(x[dev])):
        if x[dev][z] > 0:
            info.loc[((info['name'] == wait_task[z][1]) & (info['name'].index == wait_task[z][0])), 'offload'] = 1
            info.loc[((info['name'] == wait_task[z][1]) & (info['name'].index == wait_task[z][0])), 'to'] = x[dev][z]-1
            task_queue[dev][time][z][3] = 1
            task_queue[dev][time][z].append(y[dev][z])
        else:
            task_queue[dev][time][z][3] = 0
            task_queue[dev][time][z].append(0)

def offload_task_tackel(i,dev,f_off,new_t):#time,device,v,used_time
    if not wait_offload_queue[dev][i]:
            return new_t
    if new_t - i >=1:
        return i+1
    result = info.loc[(info.index == wait_offload_queue[dev][i][0][0])]
    task_need_offload = result['memory_mib'].values[0] - result['complete_offload'].values[0]
    if new_t - i == 0:
        time_slot = 1
    else:
        time_slot = 1 - (new_t - i)
    f = time_slot *max(f_off)
    if result['complete_offload'].values[0] == 0:
        info.loc[wait_offload_queue[dev][i][0][0], 'off_start'] = new_t
    if f > task_need_offload:
        info.loc[wait_offload_queue[dev][i][0][0], 'complete_offload'] = result['memory_mib'].values[0]
        info.loc[wait_offload_queue[dev][i][0][0], 'offload_success'] = 1
        loss_time = task_need_offload / f
        info.loc[wait_offload_queue[dev][i][0][0], 'off_end'] = new_t + loss_time
        #wait_size_ES
        wait_offload_queue[dev][i].pop(0)
        return new_t+loss_time
    else:
        info.loc[wait_offload_queue[dev][i][0][0], 'complete_offload'] =result['complete_offload'].values[0] + f
        return i+1

def local_task_tackel(i,dev,f_local,t1,t2):
    if not wait_tackel_queue[dev][i]:
        return [t1,t2]#CPU/GPU time
    if (t1 - i >= 1) & (t2 - i >= 1):
        return [i+1,i+1]
    result =info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1])&(info.index == wait_tackel_queue[dev][i][0][0])]
    task_need_cpu = result['cpu_milli'].values[0] - result['complete_size_cpu'].values[0]
    task_need_gpu = result['gpu_milli'].values[0] - result['complete_size_gpu'].values[0]
    if t1 - i== 0:
        time_cpu=1
    else:
        time_cpu=1-(t1 - i)
    f_cpu = f_local[0] * time_cpu
    if t2 - i== 0:
        time_gpu=1
    else:
        time_gpu=1-(t2 - i)
    f_gpu = f_local[1] * time_gpu
    if (result['complete_size_cpu'].values[0] == 0) & (result['complete_size_gpu'].values[0] == 0):
        info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1])&(
                         info.index == wait_tackel_queue[dev][i][0][0]), 'start'] = max(t1,t2)
    complete_local=[0,0]
    if f_cpu>task_need_cpu:
        info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1])&(
                         info.index == wait_tackel_queue[dev][i][0][0]), 'complete_size_cpu'] =result['complete_size_cpu'].values[0]+task_need_cpu
        info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1])&(
                         info['name'].index == wait_tackel_queue[dev][i][0][0]), 'complete_cpu'] = 1
        loss_time = task_need_cpu / f_local[0]
        time_cpu = t1 + loss_time
        complete_local[0] = 1
    else:
        info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1]) & (
                info.index == wait_tackel_queue[dev][i][0][0]), 'complete_size_cpu'] = result['complete_size_cpu'].values[0]+f_cpu
        time_cpu = i+1
    if f_gpu>task_need_gpu:
        info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1])&(
                         info.index == wait_tackel_queue[dev][i][0][0]), 'complete_size_gpu'] =result['complete_size_gpu'].values[0]+task_need_gpu
        info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1])&(
                         info.index == wait_tackel_queue[dev][i][0][0]), 'complete_gpu'] = 1
        loss_time = task_need_gpu / f_local[1]
        time_gpu = t2 + loss_time
        complete_local[1] = 1
    else:
        info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1]) & (
                info['name'].index == wait_tackel_queue[dev][i][0][0]), 'complete_size_gpu'] = result['complete_size_gpu'].values[0]+f_gpu
        time_gpu = i+1
    if all(complete_local):
        info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1])&(
                        info['name'].index == wait_tackel_queue[dev][i][0][0]), 'end'] = max(time_cpu,time_gpu)
        wait_local_size[dev][i] -= info.loc[(info['name'] == wait_tackel_queue[dev][i][0][1]) & (
                info.index == wait_tackel_queue[dev][i][0][0]), 'memory_mib'].values[0]
        wait_tackel_queue[dev][i].pop(0)
        return [max(time_cpu,time_gpu),max(time_cpu,time_gpu)]
    return [max(time_cpu,time_gpu),max(time_cpu,time_gpu)]


def critic(time,xz,yz,f_task,velocity,ind_x):
    global_update()
    MES.global_update()
    f_offload = velocity
    for i in range(time,config.get('Time') + 999):
        if i == time:
            x, y, f_mes = [], [], []
            off_index = 0
            for dev in range(config.get('Dev_dev')):
                x.append([])
                y.append([])
                f_mes.append([])
                for ind in range(len(xz[dev])):
                    x[dev].append(xz[dev][ind][ind_x])
                    y[dev].append(yz[dev][ind][ind_x])
                    f_mes[dev].append(f_task[dev][ind][ind_x])
            for dev in range(config.get('Dev_dev')):
                updage_offload_decision(i, x, y, dev)
                for task_number in range(len(task_queue[dev][i])):
                    if x[dev][task_number] > 0:
                        sub_task = info.loc[(info.index == task_queue[dev][i][task_number][0])]
                        if f_mes[dev][task_number][0] == 0:
                            return -1
                        if (sub_task['gpu_spec'].values[0] > -1) and (f_mes[dev][task_number][1] == 0):
                            return -1
                        if sub_task['gpu_spec'].values[0] == -2:
                            info.loc[info.index == task_queue[dev][i][task_number][0], 'gpu_spec'] = y[dev][task_number]
                        if sub_task['gpu_spec'].values[0] == -1:
                            info.loc[info.index == task_queue[dev][i][task_number][0], 'gpu_spec'] = 0
                        sub_task = info.loc[(info.index == task_queue[dev][i][task_number][0])]
                        if sub_task['gpu_spec'].values[0] == -1:
                            info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_cpu'] = round(
                                f_mes[dev][task_number][0] * F_es[0][0] / 12)
                            info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_gpu'] = 0
                        elif (sub_task['gpu_spec'].values[0] >= 0) or ((sub_task['gpu_spec'].values[0] == -2)):
                            info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_cpu'] = round(
                                f_mes[dev][task_number][0] * F_es[sub_task['gpu_spec'].values[0] + 1][0] / 12)
                            if f_mes[dev][task_number][1] * 4 > 1:
                                info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_gpu'] = round(
                                    f_mes[dev][task_number][1] * 4)
                            elif f_mes[dev][task_number][1] * 4 < 1 and f_mes[dev][task_number][1] * 4 > 0:
                                info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_gpu'] = \
                                f_mes[dev][task_number][1] * 4
        # ④是进入计算队列还是卸载队列
        for dev in range(config.get('Dev_dev')):
            if len(task_queue[dev][i]) == 0:
                continue
            task_queue_py = task_queue[dev][i].copy()
            for sub_task in task_queue_py:
                result = info[((info['time'] == i) & (info['time'].index == sub_task[0]))]
                if sub_task[3] == 1:
                    wait_offload_queue[dev][i].append(sub_task[:3])
                    wait_size_ES[int(result['to'].values[0])][int(result['gpu_spec'].values[0]) + 1][i] += \
                    result['memory_mib'].values[0]
                else:
                    wait_tackel_queue[dev][i].append(sub_task[:3])
                    wait_local_size[dev][i] += result['memory_mib'].values[0]  # chacha zenmezhuanlist
                task_queue[dev][i].pop(0)
        loacl_in_time, offload_in_time = np.zeros((config.get('Dev_dev'), 2)), np.zeros(config.get('Dev_dev'))
        for dev in range(config.get('Dev_dev')):
            offload_in_time[dev] = offload_task_tackel(i, dev, f_offload[dev], i)
            loacl_in_time[dev] = local_task_tackel(i, dev, f_local, i, i)
        End_user_loop(i, loacl_in_time, offload_in_time, f_local, f_offload)
        if i == time:
            fragment = MES.MES_task_tackel(i, time)
        else:
            MES.MES_task_tackel(i, time)
        result = info[(info['time'].values == time) & ((info['complete_cpu'].values == 0) | (info['complete_gpu'].values == 0))]
        if result.empty:
            break
        update_tackel_queue(i)
    return i,fragment

def End_user_loop(i,local_in_time,offload_in_time,f_local,f_offload):
    not_same_tackel =[True] * config.get('Dev_dev')
    while 1:
        The_End_tackel_available = [ True for _ in range(config.get('Dev_dev'))]
        The_End_offload_available = [True for _ in range(config.get('Dev_dev'))]
        for dev in range(config.get('Dev_dev')):
            old_local_time = local_in_time[dev].copy()
            old_offload_time = offload_in_time[dev]
            local_in_time[dev] = local_task_tackel(i, dev, f_local, local_in_time[dev][0], local_in_time[dev][1])
            if (local_in_time[dev] == [i + 1, i + 1]).all() or (local_in_time[dev] == old_local_time).all():
                The_End_tackel_available[dev] = False
            offload_in_time[dev] = offload_task_tackel(i, dev, f_offload[dev], offload_in_time[dev])
            if (offload_in_time[dev] == i+1) or (offload_in_time[dev] == old_offload_time):
                The_End_offload_available[dev] = False
            if (local_in_time[dev][0] == old_local_time[0]) &(local_in_time[dev][1] == old_local_time[1])& (offload_in_time[dev] == old_offload_time).all():
                not_same_tackel[dev] = False
        if (not any(The_End_offload_available)) & (not any(The_End_tackel_available)):
            break
    return any(not_same_tackel)

def update_tackel_queue(i):#把上一时隙没有处理完的任务队列，放入到新时隙里去
    if i == config.get('Time') + 999:
        return
    for dev in range(config.get('Dev_dev')):
        while wait_tackel_queue[dev][i]:
            if not wait_tackel_queue[dev][i]:
                break
            old_wait_task = wait_tackel_queue[dev][i].pop(0)
            wait_tackel_queue[dev][i + 1].append(old_wait_task)
        while wait_offload_queue[dev][i]:
            if not wait_offload_queue[dev][i]:
                break
            old_offload_task = wait_offload_queue[dev][i].pop(0)
            wait_offload_queue[dev][i+1].append(old_offload_task)
        wait_local_size[dev][i+1]=wait_local_size[dev][i]+wait_local_size[dev][i+1]