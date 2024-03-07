# -*- coding = utf-8 -*
# @Time：  10:32
# @File: labels.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import os

# n---0
n_path = pd.read_csv('E:/data/all_n.csv')
n = np.random.randint(0, 1, 6069)
n_df = pd.DataFrame(n, columns=['labels'])
n_data = pd.concat([n_path.iloc[:, 1:], n_df], axis=1)
n_data.to_csv('E:/data/path_labels/all_n_0.csv')

# ins---1
ins_path = pd.read_csv('E:/data/all_ins.csv')
ins = np.random.randint(1, 2, 8554)
ins_df = pd.DataFrame(ins, columns=['labels'])
ins_data = pd.concat([ins_path.iloc[:, 1:], ins_df], axis=1)
ins_data.to_csv('E:/data/path_labels/all_ins_1.csv')

# narco---2
narco_path = pd.read_csv('E:/data/all_narco.csv')
narco = np.random.randint(2, 3, 5616)
narco_df = pd.DataFrame(narco, columns=['labels'])
narco_data = pd.concat([narco_path.iloc[:, 1:], narco_df], axis=1)
narco_data.to_csv('E:/data/path_labels/all_narco_2.csv')

# plm---3
plm_path = pd.read_csv('E:/data/all_plm.csv')
plm = np.random.randint(3, 4, 7775)
plm_df = pd.DataFrame(plm, columns=['labels'])
plm_data = pd.concat([plm_path.iloc[:, 1:], plm_df], axis=1)
plm_data.to_csv('E:/data/path_labels/all_plm_3.csv')

# rbd---4
rbd_path = pd.read_csv('E:/data/all_rbd.csv')
rbd = np.random.randint(4, 5, 20918)
rbd_df = pd.DataFrame(rbd, columns=['labels'])
rbd_data = pd.concat([rbd_path.iloc[:, 1:], rbd_df], axis=1)
rbd_data.to_csv('E:/data/path_labels/all_rbd_4.csv')

# nfle---5
nfle_path = pd.read_csv('E:/data/all_nfle.csv')
nfle = np.random.randint(5, 6, 28442)
nfle_df = pd.DataFrame(nfle, columns=['labels'])
nfle_data = pd.concat([nfle_path.iloc[:, 1:], nfle_df], axis=1)
nfle_data.to_csv('E:/data/path_labels/all_nfle_5.csv')

path = 'E:/data/path_labels/'
name_list = os.listdir(path)
data = []
for x in range(len(name_list)):
    df = pd.read_csv(path+name_list[x])
    data.append(df)
data = pd.concat(data)
data.to_csv('E:/data/path_labels/all_75.csv', index=False)
