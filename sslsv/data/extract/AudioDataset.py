import os
import numpy as np

import torch
from torch.utils.data import Dataset

from sslsv.data.AudioAugmentation import AudioAugmentation

class AudioDataset(Dataset):

    def __init__(self, config):
        self.config = config
        self.load_data()

    def load_data(self):
        # Create lists of audio paths and labels
        self.files = []
        self.labels = []
        self.nb_classes = 0
        labels_id = {}
        for line in open(self.config.train):
            label, file = line.rstrip().split()

            path = os.path.join(self.config.base_path, file)
            self.files.append(path)

            if label not in labels_id:
                labels_id[label] = self.nb_classes
                self.nb_classes += 1
            self.labels.append(labels_id[label])

    def __len__(self):
        if self.config.max_samples: return self.config.max_samples
        return len(self.labels)

            
    def load_npy(self, data):
        data_ = np.load(data, allow_pickle=True)
        
        return data_


    def __getitem__(self, i):

        X = self.load_npy(self.files[i])
        if X.shape[1] > 32000:
        
            X = X[:, :32000]
        #X = np.concatenate((frame1, frame2),axis = 0)
        #X = torch.FloatTensor(X)
        y = self.labels[i]

        return X, y
