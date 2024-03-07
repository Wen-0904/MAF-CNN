# -*- coding = utf-8 -*
# @Time：  16:08
# @File: save_path.py
# @Software: PyCharm
import os
import pandas as pd
import numpy as np
from txdpy import get_letter


sbj_name = pd.read_excel('E:/cap-sleep-database-1.0.0/info.xlsx')
study_name = np.array(sbj_name['name'])
for cap_name in study_name:
    print(cap_name)
    name = get_letter(cap_name)[0]
    path1 = 'E:/data/eeg/{}'.format(cap_name)
    path2 = 'E:/data/eog/{}'.format(cap_name)
    path3 = 'E:/data/ecg/{}'.format(cap_name)
    path4 = 'E:/data/emg/{}'.format(cap_name)

    files_list1 = os.listdir(path1)
    file_path_list1 = [os.path.join(path1, i) for i in files_list1]
    files_list2 = os.listdir(path2)
    file_path_list2 = [os.path.join(path2, i) for i in files_list2]
    files_list3 = os.listdir(path3)
    file_path_list3 = [os.path.join(path3, i) for i in files_list3]
    files_list4 = os.listdir(path4)
    file_path_list4 = [os.path.join(path4, i) for i in files_list4]
    print(file_path_list1[0])
    print(file_path_list2[0])
    print(file_path_list3[0])
    print(file_path_list4[0])

    note = pd.concat([pd.DataFrame(file_path_list1, columns=['eeg']),
                      pd.DataFrame(file_path_list2, columns=['eog']),
                      pd.DataFrame(file_path_list3, columns=['ecg']),
                      pd.DataFrame(file_path_list4, columns=['emg'])], axis=1)
    note.to_csv('E:/data/{}.csv'.format(cap_name))

