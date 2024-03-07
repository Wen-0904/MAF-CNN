# -*- coding = utf-8 -*
# @Time：  18:18
# @File: test.py
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
from dataset.mydataset import MyDataset
from models.MAF_CNN import MAF_CNN
from sklearn.metrics import precision_score, \
    recall_score, f1_score, accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
import numpy as np


def my_metrics(y_ture, y_pre):
    acc = accuracy_score(y_ture, y_pre)
    precision = precision_score(y_ture, y_pre, average='macro')
    recall = recall_score(y_ture, y_pre, average='macro')
    F1 = f1_score(y_ture, y_pre, average='macro')
    kappa = cohen_kappa_score(y_ture, y_pre)
    confusion = confusion_matrix(y_ture, y_pre)
    return acc, precision, recall,  F1, kappa, confusion


device = torch.device('cpu')
cv = 10
num_classes = 6
batch_size = 256

all_y_ture = []
all_y_pre = []
start = time.time()
for k in range(cv):
    data_test = np.load('E:/data/test{}.npz'.format(k + 1))
    test_data = MyDataset(data_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('test set:{}'.format(len(test_data)))
    num_test_instances = len(test_data)

    model = MAF_CNN(num_classes)

    model_path = "E:/results/maf_cnn{}.pth".format(k + 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # test
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    all_pred = []
    all_label = []
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            data = torch.Tensor(data).to(torch.float32)
            labels = labels.to(device).to(torch.long)
            # data = data.view(-1, 1, 15360)
            eeg = data[:, 0, :].view(-1, 1, 15360)
            eog = data[:, 1, :].view(-1, 1, 15360)
            ecg = data[:, 2, :].view(-1, 1, 15360)
            emg = data[:, 3, :].view(-1, 1, 15360)
            out, outputs = model1(eeg, eog, ecg, emg)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_pred.append(predicted)
            all_label.append(labels)

        print('Test-set Accuracy of the model: {:.2f} %'.format(100 * correct / total))

        all_pred = [j.numpy() for i in all_pred for j in i]
        all_label = [j.numpy() for i in all_label for j in i]
        acc, precision, recall, F1, kappa, confusion = my_metrics(all_label, all_pred)

    all_y_ture.append(all_label)
    all_y_pre.append(all_pred)
all_y_pred = [j for i in all_y_pre for j in i]
all_y_label = [j for i in all_y_ture for j in i]
acc, precision, recall, F1, kappa, confusion = my_metrics(all_y_label, all_y_pred)
class_report = classification_report(all_y_label, all_y_pred, digits=4)
print('acc, , ,  F1, , confusion, each_acc：', acc)
print('precision：', precision)
print('recall：', recall)
print('F1 score：', F1)
print('kappa：', kappa)
print('confusion matrix ：\n', confusion)
print('classification report：\n', class_report)
