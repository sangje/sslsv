import os
import numpy as np

import torch
from torch.utils.data import Dataset
import librosa

from sslsv.data.AudioAugmentation import AudioAugmentation
from sslsv.data.utils import read_audio, load_audio

''' 오디오 파일과 프레임길이를 Input
    프레임길이만큼의 두 오디오를 Return '''
def sample_frames(audio, frame_length):
    audio_length = audio.shape[1]
    #print(audio_length)
    assert audio_length >= 2 * frame_length, \
        "audio_length should >= 2 * frame_length"

    dist = audio_length - 2 * frame_length
    dist = np.random.randint(0, dist + 1)

    lower = frame_length + dist // 2
    upper = audio_length - (frame_length + dist // 2)
    pivot = np.random.randint(lower, upper + 1)

    frame1_from = pivot - dist // 2 - frame_length
    frame1_to = pivot - dist // 2
    frame1 = audio[:, frame1_from:frame1_to]

    frame2_from = pivot + dist // 2
    frame2_to = pivot + dist // 2 + frame_length
    frame2 = audio[:, frame2_from:frame2_to]

    return frame1, frame2

class AudioDataset(Dataset):

    def __init__(self, config):
        self.config = config

        # Create augmentation module
        self.wav_augment = None
        if self.config.wav_augment.enable:
            self.wav_augment = AudioAugmentation(
                self.config.wav_augment,
                self.config.base_path
            )

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

    def preprocess_data(self, data, augment=True):
        #assert data.ndim == 2 and data.shape[0] == 1 # (1, T)
        if augment and self.wav_augment: data = self.wav_augment(data)        
        return data

    def preprocess_data_ori(self, data, augment=True):
        assert data.ndim == 2 and data.shape[0] == 1 # (1, T)
        if augment and self.wav_augment: data = self.wav_augment(data)        
        return data


    def __getitem__(self, i):
        '''

        data = read_audio(self.files[i])
        data1 = self.preprocess_data(data)
        data = data1.reshape((-1,))
        '''
        #print("shape    : ", data.shape)
        data = load_audio(
            self.files[i],
            frame_length=None,
            min_length=2*self.config.frame_length
            ) # (1, T)
        frame1, frame2 = sample_frames(data, self.config.frame_length)
        y = self.labels[i]


        
        li = []
        #print(frame1.shape)
        X = np.concatenate((
            self.preprocess_data(frame1),
            self.preprocess_data(frame2)
        ), axis=0)

        X = torch.FloatTensor(X)
        ad = self.files[i].split('/',5)
        '''
        if X.shape[1] != 32000:
            print(ad[-1], " error    size : ", X.shape[1])
        '''
        # Wav file 을 npy File로 저장하는 코드
        #np.save(self.files[i][:-3]+'npy',X)
        #print(self.files[i], "           save")

        return X, y

    
