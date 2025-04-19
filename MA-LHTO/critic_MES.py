import gv_c as gvc
import gloable_variation as gv
from config import config
import numpy as np
import importlib
import math
resource =config.get('resource')
info = gvc.info
ES_wait_queue = gvc.ES_awaiting_queue
ES_tackling_task = gvc.ES_tackling_task
wait_size_ES = gvc.wait_size_ES
ES_lasting_res = gvc.ES_lasting_res
TimeZone = gv.TimeZone
F_es=gv.F_es

def global_update():
    global info,ES_wait_queue,ES_tackling_task,wait_size_ES,ES_lasting_res
    info = gvc.info
    ES_wait_queue = gvc.ES_awaiting_queue
    ES_tackling_task = gvc.ES_tackling_task
    wait_size_ES = gvc.wait_size_ES
    ES_lasting_res = gvc.ES_lasting_res

def get_all_subtask(time,gen_task):
    task_queue=[]
    for task_number in gen_task:
        result = info.loc[(info['name'].index==task_number)&(info['time']==time)]
        if result['offload'].values[0] == 1:
            task_queue.append([task_number, result['name'].values[0], result['memory_mib'].values[0],result['off_end'].values[0]])
    return task_queue

def append_First_level(time,gen_task):
    if not gen_task:
        return
    sorted_list = sorted(gen_task, key=lambda x: x[-1])
    for task in sorted_list:
        result = info.loc[(info['name'] == task[1])&(info.index==task[0])]
        es_select = result['to'].values[0]
        #maybe update, because of Y Z not used
        ES_wait_queue[es_select][int(result['gpu_spec'].values[0])+1][time].append(task)

def get_F_use(time,es,type_):
    F_use, max_0 = [0, 0], 0
    for res in ES_lasting_res[es][type_][time]:
        F_use[0] += res[0]
        for index in range(1,5):
            if (res[index] - math.floor(res[index])) > max_0:
                max_0 = res[index] - math.floor(res[index])
            if (res[index] - math.floor(res[index])) == 0:
                F_use[1] += res[index]
    F_use[1] +=max_0
    return F_use
def select_low_fragment(time,es,t,type_):
    lengh,used=len(ES_wait_queue[es][type_][time]),-1#task number
    if lengh == 0:
        lengh=1
    F_use = get_F_use(time,es,type_)#CPU and GPU used
    fragment_value=[]
    for index_t in ES_wait_queue[es][type_][time]:
        Fr=0
        result = info.loc[(info.index == index_t[0])]
        need_cpu=result['mes_cpu'].values[0]
        need_gpu = result['mes_gpu'].values[0]
        if (F_use[0]<need_cpu) or (F_use[1]<need_gpu):
            for res in ES_lasting_res[es][type_][time]:
                for index in range(1, 5):
                    Fr += res[index]
            fragment_value.append(10000)
        if (F_use[0]>=need_cpu) and (F_use[1]>=need_gpu):
            used=0
            for res in ES_lasting_res[es][type_][time]:
                for index in range(1, 5):
                    if res[index] < min(need_gpu,1):
                        if res[index] <0:
                            print('error')
                        Fr += res[index]
            fragment_value.append(Fr/lengh)
    if not fragment_value:#the queue is null
        return -1,used,F_use
    '''
    Fr,frg = True,fragment_value[0]
    for res in fragment_value:
        if (res != frg) or (len(fragment_value) == 1): # the fragment isn't change
            Fr=False'''
    if used==-1:
        return -1, used, F_use
    used=fragment_value.index(min(fragment_value))#select the optimal task(-> minimize the fragment)
    return fragment_value[used],used,F_use

def update_wait_tackle_queue(es,type_,time,index,F_use,t):
    result = info.loc[(info.index == ES_wait_queue[es][type_][time][index][0])]
    if (F_use[0] >= result['mes_cpu'].values[0]) and (F_use[1] >= result['mes_gpu'].values[0]):
        ES_tackling_task[es][type_][time].append(ES_wait_queue[es][type_][time][index])
        ES_tackling_task[es][type_][time][-1].append(t)
        ES_wait_queue[es][type_][time].remove(ES_wait_queue[es][type_][time][index])
        delete_cpu, delete_gpu = result['mes_cpu'].values[0], result['mes_gpu'].values[0]
        for length in range(len(ES_lasting_res[es][type_][time])):
            if ES_lasting_res[es][type_][time][length][0] >= delete_cpu:
                ES_lasting_res[es][type_][time][length][0] -= delete_cpu
                delete_cpu = 0
                break
            else:
                delete_cpu = delete_cpu - ES_lasting_res[es][type_][time][length][0]
                ES_lasting_res[es][type_][time][length][0] = 0
        if delete_gpu - math.floor(delete_gpu) == 0:
            for length in range(len(ES_lasting_res[es][type_][time])):
                for gpu_n in range(1, 5):
                    if (ES_lasting_res[es][type_][time][length][gpu_n] >= delete_gpu):
                        ES_lasting_res[es][type_][time][length][gpu_n] -= delete_gpu
                        delete_gpu = 0
                        break
                    elif ES_lasting_res[es][type_][time][length][gpu_n] == 1:
                        ES_lasting_res[es][type_][time][length][gpu_n] -= 1
                        delete_gpu -= 1
                if delete_gpu == 0:
                    break
        elif (delete_gpu < 1) and (delete_gpu > 0):
            index, min_in, closest = [], 1, [None, None]
            for length in range(len(ES_lasting_res[es][type_][time])):
                for gpu_n in range(1, 5):
                    if ES_lasting_res[es][type_][time][length][gpu_n] < 1:
                        index.append([length, gpu_n, ES_lasting_res[es][type_][time][length][gpu_n]])
            index = [elem for elem in index if elem[2] > delete_gpu]
            if index:
                for element in index:
                    val = element[2] - delete_gpu
                    if val < min_in:
                        min_in = val
                        closest = [element[0], element[1]]
                ES_lasting_res[es][type_][time][closest[0]][closest[1]] -= delete_gpu
                delete_gpu = 0
            else:
                for length in range(len(ES_lasting_res[es][type_][time])):
                    for gpu_n in range(1, 5):
                        if ES_lasting_res[es][type_][time][length][gpu_n] >= delete_gpu:
                            ES_lasting_res[es][type_][time][length][gpu_n] -= delete_gpu
                            delete_gpu = 0
                            break
                    if delete_gpu == 0:
                        break

def update_processing_queue(time, es, t,type_):
    while 1:
        if type_ > 0:
            Fragment, index, F_use = select_low_fragment(time, es, t, type_)
            if index == -1:
                break
            update_wait_tackle_queue(es, type_, time, index, F_use,t)# ES_tackling_task
        else:
            to_remove = []
            for index in range(len(ES_wait_queue[es][type_][time])):
                F_use = get_F_use(time, es, type_)
                result = info.loc[(info.index == ES_wait_queue[es][type_][time][index][0])]
                if (F_use[0] >= result['mes_cpu'].values[0]):
                    ES_tackling_task[es][type_][time].append(ES_wait_queue[es][type_][time][index])
                    ES_tackling_task[es][type_][time][-1].append(t)
                    to_remove.append(index)
                    delete_cpu= result['mes_cpu'].values[0]
                    for length in range(len(ES_lasting_res[es][type_][time])):
                        if ES_lasting_res[es][type_][time][length][0] >= delete_cpu:
                            ES_lasting_res[es][type_][time][length][0] -= delete_cpu
                            delete_cpu = 0
                            break
                        else:
                            delete_cpu = delete_cpu - ES_lasting_res[es][type_][time][length][0]
                            ES_lasting_res[es][type_][time][length][0] = 0
            # 创建一个逆序排序的 to_remove 列表，以便安全地移除元素
            to_remove_sorted = sorted(to_remove, reverse=True)
            for index in to_remove_sorted:
                ES_wait_queue[es][type_][time].remove(ES_wait_queue[es][type_][time][index])
            if (not ES_wait_queue[es][type_][time]) or (not to_remove):
                break
def off_task_tackel_Fir(time, es, t,type_):  # 系统时间，第几个es，和first queue 时间
    com_time=[]
    #F_use = get_F_use(time, es, type_)
    update_processing_queue(time, es, t,type_)
    while 1:
        #F_use = get_F_use(time, es, type_)
        task_length = len(ES_tackling_task[es][type_][time])
        to_remove, complete = [], [[0, 0, 0, 0, 0] for _ in range(task_length)]
        for task_index in range(task_length):
            task = ES_tackling_task[es][type_][time][task_index]
            if task[-1] == time+1:
                continue
            result = info.loc[(info['name'] == task[1]) & (info.index == task[0])]
            task_need_cpu = result['cpu_milli'].values[0] - result['complete_size_cpu'].values[0]
            task_need_gpu = result['gpu_milli'].values[0] - result['complete_size_gpu'].values[0]
            run = info.loc[(info['name'] == task[1]) & (info.index == task[0]), 'run'].values[0]
            F_cpu, F_gpu = result['mes_cpu'].values[0], result['mes_gpu'].values[0]
            complete[task_index][3], complete[task_index][4]=F_cpu, F_gpu
            if task[-1] - time <= 1:
                F_tack = time + 1 - task[-1]  # 0.2s的
            else:
                F_tack = 0
            run_all = [0, 0]
            if (result['complete_size_cpu'].values[0] == 0) & (result['complete_size_gpu'].values[0] == 0):
                info.loc[(info['name'] == task[1]) & (info.index == task[0]), 'start'] = task[-1]
            if task_need_cpu <= F_tack * F_cpu:
                info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_size_cpu'] = \
                result['complete_size_cpu'].values[0] + task_need_cpu
                info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_cpu'] = 1
                use_time = task_need_cpu / F_cpu
                time_cpu = task[-1] + use_time
                complete[task_index][0] = 1
            else:
                info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_size_cpu'] = \
                    result['complete_size_cpu'].values[0] + F_tack * F_cpu
                use_time = F_tack
                time_cpu = task[-1] + use_time
                run_all[0] = use_time + run
                if time_cpu > time + 1:
                    time_cpu = time + 1
            if task_need_gpu <= F_tack * F_gpu:
                info.loc[(info['name'] == task[1]) & (info.index == task[0]), 'complete_size_gpu'] = \
                    result['complete_size_gpu'].values[0] + task_need_gpu
                info.loc[(info['name'] == task[1]) & (info['name'].index == task[0]), 'complete_gpu'] = 1
                if F_gpu == 0:
                    use_time=0
                else:
                    use_time = task_need_gpu / F_gpu
                time_gpu = task[-1] + use_time
                complete[task_index][1] = 1
            else:
                info.loc[(info['name'] == task[1]) & (info.index == task[0]), 'complete_size_gpu'] = \
                result['complete_size_gpu'].values[0] + F_tack * F_gpu
                use_time = F_tack
                time_gpu = task[-1] + use_time
                run_all[1] = use_time + run
                if time_gpu > time + 1:
                    time_gpu = time + 1
            if (complete[task_index][1] == 1) & (complete[task_index][0] == 1):
                info.loc[(info['name'] == task[1]) & (info.index == task[0]), 'end'] = max(time_gpu, time_cpu)
                complete[task_index][2] = max(time_gpu, time_cpu)
                to_remove.append(task_index)
                wait_size_ES[es][type_][time] = wait_size_ES[es][type_][time] - result['memory_mib'].values[0]
                com_time.append([max(time_gpu, time_cpu), 1])
            else:
                com_time.append([max(time_gpu, time_cpu), 0])
                ES_tackling_task[es][type_][time][task_index][-1] = max(time_gpu, time_cpu)
        complete = sorted(complete, key=lambda x: x[2])
        for task_index in range(task_length):
            if (complete[task_index][1] == 1) & (complete[task_index][0] == 1):
                for res_ind in range(len(ES_lasting_res[es][type_][time])):
                    if F_es[type_][0] / 2 - ES_lasting_res[es][type_][time][res_ind][0] >= complete[task_index][3]:
                        ES_lasting_res[es][type_][time][res_ind][0] += complete[task_index][3]
                        complete[task_index][3] = 0
                    else:
                        complete[task_index][3] = complete[task_index][3] - (
                                    F_es[type_][0] / 2 - ES_lasting_res[es][type_][time][res_ind][0])
                        ES_lasting_res[es][type_][time][res_ind][0] = F_es[type_][0] / 2
                    for gpu_n in range(1, 5):
                        if 1 - ES_lasting_res[es][type_][time][res_ind][gpu_n] >= complete[task_index][4]:
                            ES_lasting_res[es][type_][time][res_ind][gpu_n] += complete[task_index][4]
                            if ES_lasting_res[es][type_][time][res_ind][gpu_n]<0:
                                print('error')
                            complete[task_index][4] = 0
                            break
                        else:
                            complete[task_index][4] = complete[task_index][4] - (1 - ES_lasting_res[es][type_][time][res_ind][gpu_n])
                            ES_lasting_res[es][type_][time][res_ind][gpu_n] = 1
                update_processing_queue(time, es, complete[task_index][2], type_)
        to_remove_sorted = sorted(to_remove, reverse=True)
        for index in to_remove_sorted:
            ES_tackling_task[es][type_][time].remove(ES_tackling_task[es][type_][time][index])
        if (not ES_tackling_task[es][type_][time]) or (not to_remove):
            break
    if not com_time:
        return t
    return max(item[0] for item in com_time)

def ES_Queue_update(time):
    if time == config.get('Time') + 999:
        return
    for dev in range(config.get('Dev_edge')):
        for index in range(len(ES_lasting_res[dev])):
            wait_size_ES[dev][index][time + 1] = wait_size_ES[dev][index][time] + wait_size_ES[dev][index][time + 1]
            while ES_lasting_res[dev][index][time]:
                ES_lasting_res[dev][index][time+1].append(ES_lasting_res[dev][index][time].pop(0))
            while ES_wait_queue[dev][index][time]:
                ES_wait_queue[dev][index][time+1].append(ES_wait_queue[dev][index][time].pop(0))
            while ES_tackling_task[dev][index][time]:
                ES_tackling_task[dev][index][time+1].append(ES_tackling_task[dev][index][time].pop(0))

def get_fragment(time):
    fragment_rate,len_queue=[],[]
    for es in range(config.get('Dev_edge')):
        for type_ in range(1,config.get('resource')+1):
            lengh,used=len(ES_wait_queue[es][type_][time]),-1#task number
            len_queue.append(lengh)
            if lengh == 0:
                lengh=1
            F_use = get_F_use(time,es,type_)#CPU and GPU used
            fragment_value=[]
            for index_t in ES_wait_queue[es][type_][time]:
                Fr=0
                result = info.loc[(info.index == index_t[0])]
                need_cpu=result['mes_cpu'].values[0]
                need_gpu = result['mes_gpu'].values[0]
                if (F_use[0]<need_cpu) or (F_use[1]<need_gpu):
                    for res in ES_lasting_res[es][type_][time]:
                        for index in range(1, 5):
                            Fr += res[index]
                if (F_use[0]>=need_cpu) and (F_use[1]>=need_gpu):
                    used=0
                    for res in ES_lasting_res[es][type_][time]:
                        for index in range(1, 5):
                            if res[index] < min(need_gpu,1):
                                Fr += res[index]
                fragment_value=Fr/lengh
            if not fragment_value:#the queue is null
                fragment_value = 0
            if F_use[1]>0:
                fragment_rate.append(fragment_value/F_use[1])
            else:
                fragment_rate.append(0)
    if not fragment_rate:
        return -1
    else:
        return sum(fragment_rate)


def MES_task_tackel(i,time):
    #① 将任务信息和任务数据卸载到ES端，其中，任务信息卸载不占用时间，
    gen_task,fragement = TimeZone[i],0#保存的是id号
    gen_task = get_all_subtask(i, gen_task)
    # ② 就是将每个时隙内的任务信息首先依次输入到First_level_queue中，因为排队处理嘛，有信息就可以排队了
    append_First_level(i, gen_task)
    if i == time:
        fragement = get_fragment(i)
    t = [[i]*(config.get("resource")+1) for _ in range(config.get('Dev_edge'))]
    for dev in range(config.get('Dev_edge')):
        for type in range(resource+1):
            if not ES_wait_queue[dev][type][i]:
                t[dev][type] = i
            elif ES_wait_queue[dev][type][i][0][3]>i:
                t[dev][type] = ES_wait_queue[dev][type][i][0][3]

    # ③ 每个时隙对队首的任务进行判断，如果任务尚未传输完毕/任务的前置任务没有完成，则让任务进入等待队列进行等待
    First_time = np.zeros((config.get('Dev_edge'), resource+1))
    for dev in range(config.get('Dev_edge')):
        # 用于得到每个列表的起始时间
        for type in range(resource+1):
            First_time[dev][type] = off_task_tackel_Fir(i, dev, t[dev][type], type)
    ES_Queue_update(i)
    if i == time:
        return fragement
    else:
        return -1