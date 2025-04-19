import importlib
import random
import numpy as np
from tqdm import trange
import math
import pandas as pd
from config import config
import gloable_variation as gv
import OnlineGPU_MES as MES
import Critic_module as Critic
from memory import multiagent
from memory import unit_all

curPath = config.get("curPath")
Datasets_path = config.get("Datasets_path")
epoch_all = config.get("epoch")
Model_path = config.get("Model_path")
Result_path = config.get("Result_path")
alpha = config.get("alpha")
save_path = config.get("Datasets_path")

def append_to_task_queue(time,gen_task):
    if len(gen_task) == 0:
        return
    for id in gen_task:
        task_name=info.loc[id]['name']
        from_=int(info.loc[id]['from'])
        reslut = info[(info['name'].index == id)]
        task_size = reslut['memory_mib'].values[0]
        task=[id,task_name,task_size,0]
        task_queue[from_][time].append(task)
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

def get_user_para(time,user,task_number):
    #the_first
    ones_time = [time/(config.get('Time')+1)]
    #the second
    ones_task = []
    task_dicision = task_queue[user][time][task_number]
    result = info[((info['time'] == time) & (info.index == task_dicision[0]))]
    if result['memory_mib'].values[0]>100000:
        ones_task.append(1)
    else:
        ones_task.append(result['memory_mib'].values[0] / 100000)
    if result['cpu_milli'].values[0] >50000:
        ones_task.append(1)
    else:
        ones_task.append(result['cpu_milli'].values[0]/50000)
    if result['gpu_milli'].values[0] >10:
        ones_task.append(1)
    else:
        ones_task.append(result['gpu_milli'].values[0]/10)
    ones_task.append((result['qos'].values[0]) / 3)
    ones_task.append((result['gpu_spec'].values[0]+1)/3)
    #the third
    ones_other=[]
    result_others = list(filter(lambda x: x != task_dicision, task_queue[user][time]))
    number_other_task,user_size,size_cpu,size_gpu,priorty_lx=0,0,0,0,[0,0,0]
    result1 = info[((info['time'] == time) & (info['time'].index == task_dicision[0]))]
    for task_other in result_others:
        result = info[((info['time'] == time) & (info['time'].index == task_other[0]))]
        if (result['gpu_spec'].values[0] == -1) or (result['gpu_spec'].values[0] == 0):
            number_other_task +=1
            user_size += result['memory_mib'].values[0]
            size_cpu += result['cpu_milli'].values[0]
            if result1['gpu_spec'].values[0] == 0:
                size_gpu += result['gpu_milli'].values[0]
            priorty_lx[result['qos'].values[0]-1] +=1

    priorty_lx = [x/3 for x in priorty_lx]
    ones_other.append(number_other_task/5)
    ones_other.append(user_size / 500000)
    ones_other.append(size_cpu/250000)
    ones_other.append(size_gpu/50)
    ones_other = ones_other+priorty_lx
    #the forth:user's ability  2
    user_size,user_number_task=0,0
    for task_other in wait_tackel_queue[user][time]:
        result = info[(info.index == task_other[0])]
        user_size += result['memory_mib'].values[0]/50000
        user_number_task += 1
    ones_user = [user_size/2500000,user_number_task/50]
    # the forth
    mes_task_wait, task_need_cpu, task_need_gpu, mes_size_cpu, mes_size_gpu = np.zeros(
        (M, config.get('resource') + 1)), np.zeros((M, config.get('resource') + 1)), np.zeros(
        (M, config.get('resource') + 1)), np.zeros((M, config.get('resource') + 1)), np.zeros(
        (M, config.get('resource') + 1))
    for mes in range(M):
        for gepc in range(config.get('resource') + 1):
            for task in ES_awaiting_queue[mes][gepc][time]:
                result_others = info[(info.index == task[0])]
                mes_task_wait[mes][gepc] += 1
                task_need_cpu[mes][gepc] += result_others['cpu_milli'].values[0]
                task_need_gpu[mes][gepc] += result_others['gpu_milli'].values[0]
                mes_size_cpu[mes][gepc] += result_others['mes_cpu'].values[0]
                mes_size_gpu[mes][gepc] += result_others['mes_gpu'].values[0]
            if mes_task_wait[mes][gepc] > 0:
                task_need_cpu[mes][gepc] = task_need_cpu[mes][gepc] / (mes_task_wait[mes][gepc] * 50000)
                task_need_gpu[mes][gepc] = task_need_gpu[mes][gepc] / (mes_task_wait[mes][gepc] * 10)
                mes_size_cpu[mes][gepc] = mes_size_cpu[mes][gepc] / (mes_task_wait[mes][gepc] * F_es[gepc][0])
                if F_es[gepc][1] > 0:
                    mes_size_gpu[mes][gepc] = mes_size_gpu[mes][gepc] / (mes_task_wait[mes][gepc] * F_es[gepc][1])
                else:
                    mes_size_gpu[mes][gepc] = 0
                mes_task_wait[mes][gepc] = mes_task_wait[mes][gepc] / 100
    ones_mes = mes_task_wait.reshape(-1).tolist() + task_need_cpu.reshape(-1).tolist() + task_need_gpu.reshape(
        -1).tolist() + mes_size_cpu.reshape(-1).tolist() + mes_size_gpu.reshape(-1).tolist()
    ones_mes = Guiyi.get_output_mes(ones_mes)
    return ones_time+ones_task+ones_other+ones_user+ones_mes.tolist()#1+5+7

def save_file(ephoch,info,wait_size_Vehile,wait_size_ES,value_all,All_fragment):
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_"+str(ephoch)+ "_" + '5' + ".csv"
    info.to_csv(save_name, index=None)
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_"+str(ephoch)+ "_" + 'EU_wait_size' + ".csv"
    wait_size_Vehile = pd.DataFrame(wait_size_Vehile)
    wait_size_Vehile.to_csv(save_name, index=False, header=False)
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_"+str(ephoch)+ "_" + 'MES_wait_size' + ".csv"
    Wait_size=[]
    for es in range(M):
        Wait_size.append([])
        for time in range(config.get('Time') + 1000):
            Wait_size[-1].append(wait_size_ES[es][0][time]+wait_size_ES[es][1][time]+wait_size_ES[es][2][time])
    wait_size_ES = pd.DataFrame(Wait_size)
    wait_size_ES.to_csv(save_name, index=False, header=False)
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_" + str(ephoch) + "_" + 'value_all' + ".csv"
    value_all = pd.DataFrame(value_all)
    value_all.to_csv(save_name, index=False, header=False)
    save_name = curPath + save_path + "info_EU_" + str(config.get("Time")) + "_" + str(ephoch) + "_" + 'All_fragment' + ".csv"
    All_fragment = pd.DataFrame(All_fragment)
    All_fragment.to_csv(save_name, index=False, header=False)

if __name__ == '__main__':
    N = config.get("Dev_dev")
    M = config.get('Dev_edge')
    Band = config.get("B")
    P_tran_min = config.get('P_tran_min')
    P_tran_max = config.get('P_tran_max')
    alpha_1 = config.get('alpha_1')
    coverage = config.get('Scope')
    vehicle_position = config.get("vehicle_position")
    edge_position = config.get("Edge_position")
    noisy = config.get("noisy")
    F_es = gv.F_es
    ES_lasting_res = gv.ES_lasting_res
    loss_exponent = 3
    light = 3 * 10 ** 8  # 光速
    Ad = 3  # 接收天线增益？
    fc = 915 * 10 ** 6  # frequen
    
    for epoch in range(epoch_all):
        multi_user = multiagent(config.get('Dev_dev'), 26, M+1, config.get('resource'), 2)
        Guiyi = unit_all(M*(config.get('resource')+1)*5, 11)
        info = gv.info
        TimeZone = gv.TimeZone
        f_local = gv.F_eu
        task_queue = gv.task_queue  # device
        wait_tackel_queue = gv.wait_tackel_queue  # device
        wait_local_size = gv.wait_local_size
        wait_offload_queue = gv.wait_offload_queue  # device
        wait_size_ES = gv.ES_wait_size
        ES_awaiting_queue= gv.ES_awaiting_queue
        All_reward = np.zeros(config.get('Time') + 1000)
        All_fragment = np.zeros(config.get('Time') + 1000)
        for i in trange(config.get('Time') + 1000):
            h0 = np.zeros((config.get('Dev_dev'), config.get('Dev_edge')))
            dist_v = np.zeros((config.get('Dev_dev'), config.get('Dev_edge')))
            velocity = np.zeros((config.get('Dev_dev'), config.get('Dev_edge')))
            for dev in range(config.get('Dev_dev')):
                for edge in range(config.get('Dev_edge')):
                    dist_v[dev][edge] =math.sqrt((vehicle_position[dev][0] - edge_position[edge][0])**2+(vehicle_position[dev][1] - edge_position[edge][1])**2)
            dist_v_flatten =dist_v.flatten()
            for j in range(config.get('Dev_dev')):  # 计算信道系数的公式为d*(light/4/math.pi/fc/dist_v[j])**(loss_exponent)
                for k in range(config.get('Dev_edge')):
                    h0[j][k] = Ad * (light / 4 / math.pi / fc / dist_v[j][k]) ** (loss_exponent)
            gen_task = TimeZone[i]
            append_to_task_queue(i, gen_task)
            for j in range(config.get('Dev_dev')):
                for k in range(config.get('Dev_edge')):
                    velocity[j][k] = Band * math.log2(
                        1 + ((h0[j][k] * config.get('P_tran_max')) / (10 ** ((noisy - 30) / 10))))  # 卸载速率
            ### 进行任务处置
            if i < config.get('Time'):
                if i < 50:
                    K = 3
                elif i < 100:
                    K = 3
                else:
                    K = 3
                xz,yz,fz,input_user = [],[],[],[]
                for n in range(N):
                    xz.append([])
                    yz.append([])
                    fz.append([])
                    input_user.append([])
                    for task_number in range(len(task_queue[n][i])):# 或1
                        result = info[((info['time'] == i) & (info.index == task_queue[n][i][task_number][0]))]
                        single_task = get_user_para(i, n, task_number)
                        input_user[-1].append(single_task)
                        x=multi_user.choose_action_xy(n,single_task)
                        indexed_Z = list(enumerate(x))
                        # 按值从大到小排序
                        sorted_Z = sorted(indexed_Z, key=lambda k: k[1], reverse=True)
                        # 获取前k个最大值的索引
                        x = [sorted_Z[i][0] for i in range(K)]
                        xz[-1].append(x)
                        yz[-1].append([])
                        fz[-1].append([])
                        for ind in range(len(x)):
                            if (x[ind] > 0) and (result['gpu_spec'].values[0] == -2):
                                z = multi_user.choose_action_z(n, single_task)
                                z = z.index(max(z))
                            else:
                                z = result['gpu_spec'].values[0]
                            yz[n][-1].append(z)
                            f = multi_user.choose_action_f(n, single_task, i)
                            fz[n][-1].append(f)
                        #x = 1
                ind_x, All_reward[i] = Critic.critic_en(i, xz, yz, fz, velocity, K)
                for dev in range(N):
                    for task_number in range(len(xz[dev])):
                        multi_user.remember_Dev1(dev,input_user[dev][task_number],xz[dev][task_number][ind_x],All_reward[i])
                        if xz[dev][task_number][ind_x] > 0:
                            if yz[dev][task_number][ind_x]>=0:
                                multi_user.remember_Dev2(dev,input_user[dev][task_number],yz[dev][task_number][ind_x],All_reward[i])
                            if (fz[dev][task_number][ind_x][0]>0) and (fz[dev][task_number][ind_x][1]>0):
                                multi_user.remember_Dev3(dev, input_user[dev][task_number], fz[dev][task_number][ind_x],All_reward[i])
                x, y, f_mes = [], [], []
                for dev in range(config.get('Dev_dev')):
                    x.append([])
                    y.append([])
                    f_mes.append([])
                    for ind in range(len(xz[dev])):
                        x[dev].append(xz[dev][ind][ind_x])
                        y[dev].append(yz[dev][ind][ind_x])
                        f_mes[dev].append(fz[dev][ind][ind_x])
                for dev in range(config.get('Dev_dev')):
                    updage_offload_decision(i, x, y, dev)
                    for task_number in range(len(task_queue[dev][i])):
                        if x[dev][task_number] > 0:
                            sub_task = info.loc[(info.index == task_queue[dev][i][task_number][0])]
                            if sub_task['gpu_spec'].values[0] == -2:
                                info.loc[info.index == task_queue[dev][i][task_number][0], 'gpu_spec'] = y[dev][
                                    task_number]
                            if sub_task['gpu_spec'].values[0] == -1:
                                info.loc[info.index == task_queue[dev][i][task_number][0], 'gpu_spec'] = 0
                            sub_task = info.loc[(info.index == task_queue[dev][i][task_number][0])]
                            if sub_task['gpu_spec'].values[0] == -1:
                                info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_cpu'] = \
                                f_mes[dev][task_number][0] * F_es[0][0]
                                info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_gpu'] = 0
                            elif (sub_task['gpu_spec'].values[0] >= 0) or ((sub_task['gpu_spec'].values[0] == -2)):
                                info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_cpu'] = \
                                f_mes[dev][task_number][0] * F_es[sub_task['gpu_spec'].values[0] + 1][0]
                                info.loc[info.index == task_queue[dev][i][task_number][0], 'mes_gpu'] = \
                                f_mes[dev][task_number][1] * 4
            # ④是进入计算队列还是卸载队列
            f_offload = velocity
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
            End_loop_judge = End_user_loop(i, loacl_in_time, offload_in_time, f_local, f_offload)
            ##############from here, the above is local computing, the last is ES computing
            All_fragment[i] = MES.MES_task_tackel(i)
            update_tackel_queue(i)
        save_file(epoch, info, wait_local_size, wait_size_ES, All_reward, All_fragment)
        importlib.reload(gv)
        importlib.reload(MES)
        '''

                for dev in range(config.get('Dev_dev')):
                    for task_number in range(len(xz_sigmod[dev])):
                        multi_user.remember_Dev(dev, input_user[dev][task_number], xz_sigmod[dev][task_number],1)
                for task_number in range((len(yz_sigmod))):
                    multi_user.remember_BS(input_mes[task_number],yz_sigmod[task_number])
        '''