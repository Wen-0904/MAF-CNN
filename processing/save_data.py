# -*- coding = utf-8 -*
# @Time：  15:19
# @File: sava_data.py
# @Software: PyCharm
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from my_dataset.dataset.mydataset import MyDataset
from torch.utils.data import DataLoader
from collections import Counter

if __name__ == '__main__':
    cv = 10
    for k in range(cv):
        print('E:/data/path_labels/all_{}.csv'.format(k+1))
        test_path = pd.read_csv('E:/data/path_labels/all_{}.csv'.format(k + 1))
        eeg = []
        eog = []
        ecg = []
        emg = []
        for i in tqdm(test_path['eeg']):
            da1 = np.loadtxt(i).astype(np.float32)
            eeg.append(da1)
        for i in tqdm(test_path['eog']):
            da2 = np.loadtxt(i).astype(np.float32)
            eog.append(da2)
        for i in tqdm(test_path['ecg']):
            da3 = np.loadtxt(i).astype(np.float32)
            ecg.append(da3)
        for i in tqdm(test_path['emg']):
            da4 = np.loadtxt(i).astype(np.float32)
            emg.append(da4)
        np.savez('E:/data/test{}.npz'.format(k + 1), eeg=np.array(eeg), eog=np.array(eog), ecg=np.array(ecg),
                 emg=np.array(emg), labels=np.array(test_path['labels']))
        print('all test data label %s' % Counter(test_path['labels']))

        eeg1 = []
        eog1 = []
        ecg1 = []
        emg1 = []
        path = pd.DataFrame()
        for m in range(1, cv+1):
            if m != (k + 1):
                print('E:/data/path_labels/all_{}.csv'.format(m))
                p = pd.read_csv('E:/data/path_labels/all_{}.csv'.format(m))
                path = pd.concat([path, p])

        train_path, val_path, y_train, y_val = train_test_split(path, path['labels'], random_state=42,
                                                                test_size=0.1, stratify=path['labels'])

        for i in tqdm(train_path['eeg']):
            da1 = np.loadtxt(i).astype(np.float32)
            eeg1.append(da1)
        for i in tqdm(train_path['eog']):
            da2 = np.loadtxt(i).astype(np.float32)
            eog1.append(da2)
        for i in tqdm(train_path['ecg']):
            da3 = np.loadtxt(i).astype(np.float32)
            ecg1.append(da3)
        for i in tqdm(train_path['emg']):
            da4 = np.loadtxt(i).astype(np.float32)
            emg1.append(da4)
        np.savez('E:/data/train{}.npz'.format(k + 1), eeg=np.array(eeg1), eog=np.array(eog1), ecg=np.array(ecg1),
                 emg=np.array(emg1), labels=np.array(train_path['labels']))
        print('all train data label %s' % Counter(train_path['labels']))

        eeg2 = []
        eog2 = []
        ecg2 = []
        emg2 = []

        for ii in tqdm(val_path['eeg']):
            da1 = np.loadtxt(ii).astype(np.float32)
            eeg2.append(da1)
        for ii in tqdm(val_path['eog']):
            da2 = np.loadtxt(ii).astype(np.float32)
            eog2.append(da2)
        for ii in tqdm(val_path['ecg']):
            da3 = np.loadtxt(ii).astype(np.float32)
            ecg2.append(da3)
        for ii in tqdm(val_path['emg']):
            da4 = np.loadtxt(ii).astype(np.float32)
            emg2.append(da4)
        np.savez('E:/data/val{}.npz'.format(k + 1), eeg=np.array(eeg2), eog=np.array(eog2), ecg=np.array(ecg2),
                 emg=np.array(emg2), labels=np.array(val_path['labels']))
        print('all validate data label %s' % Counter(val_path['labels']))
