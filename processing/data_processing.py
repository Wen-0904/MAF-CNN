# -*- coding = utf-8 -*
# @Time：  10:31
# @File: data_processing.py
# @Software: PyCharm
import numpy as np
from mne.io import read_raw_edf

if __name__ == '__main__':

    raw = read_raw_edf('E:/cap-sleep-database-1.0.0/n1.edf')
    fs = 512
    data = raw.to_data_frame()[fs * 30:]
    eeg = data['C4-A1']
    # eog = data['ROC-LOC']
    # ecg = data['ECG1-ECG2']
    # emg = data['EMG1-EMG2']

    all_eeg_30s = []
    for i in range(int(len(eog) / fs / 30)):
        eeg_30s = eog[30 * fs * i:30 * fs * (i + 1)].tolist()
        all_eeg_30s.append(eeg_30s)
    all_eeg_30s = np.array(all_eeg_30s)
    # save data
    path = 'E:/data/eeg/n1'
    for j in range(len(all_eeg_30s)):
        s = '%04d' % j
        print(s)
        np.savetxt(path + '/%s' % str(s) + '.txt', all_eeg_30s[j])







