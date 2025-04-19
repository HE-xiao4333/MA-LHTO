import copy
import gloable_variation as gv


info = copy.deepcopy(gv.info)
task_queue = copy.deepcopy(gv.task_queue)
wait_tackel_queue = copy.deepcopy(gv.wait_tackel_queue)
wait_offload_queue = copy.deepcopy(gv.wait_offload_queue)
wait_local_size = copy.deepcopy(gv.wait_local_size)
wait_size_ES = copy.deepcopy(gv.ES_wait_size)
ES_awaiting_queue = copy.deepcopy(gv.ES_awaiting_queue)
ES_tackling_task = copy.deepcopy(gv.ES_tackling_task)
ES_lasting_res = copy.deepcopy(gv.ES_lasting_res)
f_local = gv.F_eu
F_es = gv.F_es

