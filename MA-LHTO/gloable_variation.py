import numpy as np
from config import config
import pandas as pd

curPath = config.get("curPath")
Datasets_path = config.get("Datasets_path")
epoch_all = config.get("epoch")
Model_path = config.get("Model_path")
Result_path = config.get("Result_path")
#alpha = config.get("alpha")
save_path = config.get("Datasets_path")

file_info = curPath + Datasets_path + "info_" + str(config.get("Time")) +"_"+str(config.get("Dev_dev"))+".csv"
info = pd.read_csv(file_info)  # 读任务文件
file_es = curPath + Datasets_path + "info_GPU.csv"
info_GPU = pd.read_csv(file_es)  # 读任务文件

task_queue = []  # ID
wait_tackel_queue = []  # ID
wait_offload_queue = []  # ID
wait_local_size = []  # ID
for _ in range(config.get('Dev_dev')):
    task_queue.append([])
    wait_tackel_queue.append([])
    wait_offload_queue.append([])
    wait_local_size.append([])
for j in range(config.get('Dev_dev')):
    for i in range(config.get('Time')+1000):
        task_queue[j].append([])
        wait_tackel_queue[j].append([])
        wait_offload_queue[j].append([])
        wait_local_size[j].append(0)

F_eu=[96000,4] #CPU,GPU, G2 96000 786432
#######################################################################################################
#'G3','V100M32','G2','V100M16','T4'
ES_wait_size=[]
ES_awaiting_queue=[]
ES_tackling_task=[]
ES_lasting_res = []
for _ in range(config.get("Dev_edge")):
    ES_wait_size.append([])
    ES_awaiting_queue.append([])
    ES_tackling_task.append([])
    ES_lasting_res.append([])
    for _ in range(3):
        ES_wait_size[-1].append([])
        ES_awaiting_queue[-1].append([])
        ES_tackling_task[-1].append([])
        ES_lasting_res[-1].append([])

for _ in range(config.get('Time')+1000):
    for j in range(config.get('Dev_edge')):
        for k in range(3):
            ES_wait_size[j][k].append(0)
            ES_awaiting_queue[j][k].append([])
            ES_tackling_task[j][k].append([])
            ES_lasting_res[j][k].append([])

F_es=[[96000*3,0]]
for i in range(2):
    F_es.append([(info_GPU.iloc[i]['cpu_milli']*3),(info_GPU.iloc[i]['gpu']*3)])

for edge in range(config.get('Dev_edge')):
    ES_lasting_res[edge][0][0].append([96000, 0,0,0,0])
    ES_lasting_res[edge][0][0].append([96000, 0,0,0,0])
    ES_lasting_res[edge][0][0].append([96000, 0, 0, 0, 0])
    for k in range(2):
        ES_lasting_res[edge][k+1][0].append([info_GPU.iloc[k]['cpu_milli'], 1,1,1,1])
        ES_lasting_res[edge][k+1][0].append([info_GPU.iloc[k]['cpu_milli'], 1,1,1,1])
        ES_lasting_res[edge][k+1][0].append([info_GPU.iloc[k]['cpu_milli'], 1, 1, 1, 1])




###########################################################################################################################
TimeZone = []
for i in range(config.get("Time") + 1000):
    temp = info.loc[info['time'] == i].index.tolist()
    TimeZone.append(temp)



#all string to save
#NL para save
model_save_RSU = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "_1_" + "model_RSU.pt"
model_save_VE = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "_1_"
model_save_LSTM = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "LSTM.pt"
Guiyi_RSU = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "Allin_RSU.pt"
Guiyi_VE = curPath + save_path + "model_" + str(config.get("Time")) + "_" + str(config.get("Dev_dev")) + "Allin_VE.pt"