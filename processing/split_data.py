# -*- coding = utf-8 -*
# @Time：  15:39
# @File: split_data.py
# @Software: PyCharm
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('E:/data/path_labels/all_75.csv')

x_train, x_test1, y_train, y_test = train_test_split(df, df['labels'], shuffle=True,
                                                     test_size=1/10, stratify=df['labels'])
# 1 fold
x_test1.to_csv('E:/data/path_labels/all_1.csv')
x_train1, x_test2, y_train1, y_test1 = train_test_split(x_train, x_train['labels'], shuffle=True,
                                                        test_size=1/9, stratify=x_train['labels'])
# 2 fold
x_test2.to_csv('E:/data/path_labels/all_2.csv')
x_train2, x_test3, y_train2, y_test2 = train_test_split(x_train1, x_train1['labels'], shuffle=True,
                                                        test_size=1/8, stratify=x_train1['labels'])
# 3 fold
x_test3.to_csv('E:/data/path_labels/all_3.csv')
x_train3, x_test4, y_train3, y_test3 = train_test_split(x_train2, x_train2['labels'], shuffle=True,
                                                        test_size=1/7, stratify=x_train2['labels'])
# 4 fold
x_test4.to_csv('E:/data/path_labels/all_4.csv')
x_train4, x_test5, y_train4, y_test4 = train_test_split(x_train3, x_train3['labels'], shuffle=True,
                                                        test_size=1/6, stratify=x_train3['labels'])
# 5 fold
x_test5.to_csv('E:/data/path_labels/all_5.csv')
x_train5, x_test6, y_train5, y_test5 = train_test_split(x_train4, x_train4['labels'], shuffle=True,
                                                        test_size=1/5, stratify=x_train4['labels'])
# 6 fold
x_test6.to_csv('E:/data/path_labels/all_6.csv')
x_train6, x_test7, y_train6, y_test6 = train_test_split(x_train5, x_train5['labels'], shuffle=True,
                                                        test_size=1/4, stratify=x_train5['labels'])
# 7 fold
x_test7.to_csv('E:/data/path_labels/all_7.csv')
x_train7, x_test8, y_train7, y_test7 = train_test_split(x_train6, x_train6['labels'], shuffle=True,
                                                        test_size=1/3, stratify=x_train6['labels'])
# 8 fold
x_test8.to_csv('E:/data/path_labels/all_8.csv')
x_train8, x_test9, y_train8, y_test8 = train_test_split(x_train7, x_train7['labels'], shuffle=True,
                                                        test_size=1/2, stratify=x_train7['labels'])
# 9 fold
x_test9.to_csv('E:/data/path_labels/all_9.csv')
# 10 fold
x_train8.to_csv('E:/data/path_labels/all_10.csv')


