# -*- coding = utf-8 -*
# @Time：  17:11
# @File: mydataset.py
# @Software: PyCharm

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):

    def __init__(self, data):
        eeg = data['eeg']
        eog = data['eog']
        ecg = data['ecg']
        emg = data['emg']
        data1 = []
        for i in range(len(eeg)):
            all_data = np.array([eeg[i], eog[i], ecg[i], emg[i]])
            data1.append(all_data)
        self.data = torch.from_numpy(np.array(data1))
        label1 = data['labels']
        self.label = torch.from_numpy(np.array(label1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label
