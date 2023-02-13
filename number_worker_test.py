import torch
from torch.utils.data import DataLoader
import torchaudio
import time

yesno_data = torchaudio.datasets.YESNO('.', download=False)

def collate_fn(batch):

    tensors = [b[0].t() for b in batch if b]
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    tensors = tensors.transpose(1, -1)

    targets = torch.tensor([b[2] for b in batch if b])
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return tensors, targets

pin_memory = False
print('pin_memory is', pin_memory)
for num_workers in range(0, 20, 1): 
    data_loader = torch.utils.data.DataLoader(
                                                yesno_data,
                                                batch_size=16,
                                                pin_memory=pin_memory,
                                                num_workers=num_workers,
                                                collate_fn=collate_fn)
    start = time.time()
    for epoch in range(1, 5):
        for i, (data, _) in enumerate(data_loader):
            torch.tensor(data,'cuda')
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

pin_memory = True
print('pin_memory is', pin_memory)
for num_workers in range(0, 20, 1): 
    data_loader = torch.utils.data.DataLoader(
                                                yesno_data,
                                                batch_size=16,
                                                pin_memory=pin_memory,
                                                num_workers=num_workers,
                                                collate_fn=collate_fn)
    start = time.time()
    for epoch in range(1, 5):
        for i, (data, _) in enumerate(data_loader):
            torch.tensor(data,'cuda')
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))