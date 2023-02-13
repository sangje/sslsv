import torch
from torch.utils.data import DataLoader
import torchaudio
from time import time

yesno_data = torchaudio.datasets.YESNO('.', download=False)


pin_memory = False
print('pin_memory is', pin_memory)
for num_workers in range(0, 20, 1): 
    data_loader = torch.utils.data.DataLoader(
                                                yesno_data,
                                                batch_size=1,
                                                pin_memory=pin_memory,
                                                num_workers=num_workers)
    start = time.time()
    for epoch in range(1, 5):
        for i, data in enumerate(data_loader, 0):
            pass
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

pin_memory = True
print('pin_memory is', pin_memory)
for num_workers in range(0, 20, 1): 
    data_loader = torch.utils.data.DataLoader(
                                                yesno_data,
                                                batch_size=1,
                                                pin_memory=pin_memory,
                                                num_workers=num_workers)
    start = time.time()
    for epoch in range(1, 5):
        for i, data in enumerate(data_loader, 0):
            pass
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))