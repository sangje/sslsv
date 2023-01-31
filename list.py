import os
import numpy as np

trial_list = "./data/trials"

with open(trial_list) as dataset_file:
    total_vox2 = dataset_file.readlines()
    

a = []

for to in total_vox2:

  target, a, b = to.split(' ')
  if a.startswith('Vox'):
    pass
  else:
    a.append(to)


with open("/media/nextgen/Samsung_T5/sslsv/data/vox1_trials",'w',encoding='UTF-8') as f:
    for k in range(len(a)):
        #line = a[k] + '\n'
        line = a[k]
        f.write(line)
