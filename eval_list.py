import os
import numpy as np


trial_list = "./data/trials"

with open(trial_list) as dataset_file:
    total_vox2 = dataset_file.readlines()

to = []

for lidx, line in enumerate(total_vox2):
    data = line.strip().split()
    id1 = data[1].split('/',3)[0]
    id2 = data[2].split('/',3)[0]

    to.append(id1+ " voxceleb1/" + data[1])
    to.append(id2+ " voxceleb1/" + data[2])
    
    
a = list(set(to))

with open("/media/nextgen/Samsung_T5/sslsv/data/eval_list.txt",'w',encoding='UTF-8') as f:
    for k in range(len(a)):
        line = a[k] + '\n'
        f.write(line)
