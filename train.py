# -*- coding = utf-8 -*
# @Time：  17:07
# @File: train.py
# @Software: PyCharm

from torch.utils.data import DataLoader
from models.MAF_CNN import MAF_CNN
from dataset.mydataset import MyDataset
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# hyperparameters
learning_rate = 0.0001
num_epochs = 100
batch_size = 256

cv = 10
num_classes = 6

# 10-fold cross-validation
for k in range(cv):
    # data
    data_train = np.load('E:/data/train{}.npz'.format(k + 1))
    data_val = np.load('E:/data/val{}.npz'.format(k + 1))
    train_data = MyDataset(data_train)
    val_data = MyDataset(data_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    print('training set:{}\nvalidation set:{}'.format(len(train_data), len(val_data)))

    num_train_instances = len(train_data)
    num_val_instances = len(val_data)
    num_train_batch = len(train_loader)
    num_val_batch = len(val_loader)

    model = MAFCNN(num_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001)
    train_loss = np.zeros([num_epochs, 1])
    val_loss = np.zeros([num_epochs, 1])
    train_acc = np.zeros([num_epochs, 1])
    val_acc = np.zeros([num_epochs, 1])

    best_acc, best_epoch = 0, 0
    start_time = time.time()
    # train
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        loss_all = 0
        correct_train = 0
        for step, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            data = torch.Tensor(data).to(torch.float32)
            labels = labels.to(device).to(torch.long)
            eeg = data[:, 0, :].view(-1, 1, 15360)
            eog = data[:, 1, :].view(-1, 1, 15360)
            ecg = data[:, 2, :].view(-1, 1, 15360)
            emg = data[:, 3, :].view(-1, 1, 15360)
            # labels = labels.view(-1, 1)
            # Forward pass
            out, outputs = model(eeg, eog, ecg, emg)
            loss = loss_fn(outputs, labels)
            # loss = focal_loss(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct_train += (predicted == labels).sum().item()
            # acc = (predicted == labels).float().mean()

            if not step % 100:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
                      % (epoch + 1, num_epochs, step, total_step, loss))
        train_loss[epoch] = loss_all / num_train_batch
        print('Epoch: %03d/%03d Training accuracy: %.2f%% Training loss: %.4f' % (
            epoch + 1, num_epochs,  (100 * float(correct_train) / num_train_instances), loss_all / num_train_batch))
        train_acc[epoch] = (100 * float(correct_train) / num_train_instances)

        model.eval()
        with torch.no_grad():
            loss_all = 0
            correct_val = 0
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                data = torch.Tensor(data).to(torch.float32)
                labels = labels.to(device).to(torch.long)
                eeg = data[:, 0, :].view(-1, 1, 15360)
                eog = data[:, 1, :].view(-1, 1, 15360)
                ecg = data[:, 2, :].view(-1, 1, 15360)
                emg = data[:, 3, :].view(-1, 1, 15360)
                # labels = labels.view(-1, 1)
                # Forward pass
                out, outputs = model(eeg, eog, ecg, emg)
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

                loss = loss_fn(outputs, labels)
                loss_all += loss.item()
            avg_val_accuracy = 100 * float(correct_val) / num_val_instances
            avg_val_loss = loss_all / num_val_batch
            print('Val accuracy: %.2f%%  Val loss: %.4f' % (avg_val_accuracy, avg_val_loss))
            val_loss[epoch] = avg_val_loss
            val_acc[epoch] = avg_val_accuracy

        # best accuracy
        if avg_val_accuracy > best_acc:
            best_epoch = epoch
            best_acc = avg_val_accuracy
            print('best_acc:', best_acc, 'best_epoch:', best_epoch+1)
            print("save model：")
            torch.save(model.state_dict(), "E:/results/maf_cnn{}.pth"
                       .format(k + 1))

    result = pd.concat([pd.DataFrame(train_acc, columns=['train_acc']),
                        pd.DataFrame(train_loss, columns=['train_loss']),
                        pd.DataFrame(val_acc, columns=['val_acc']),
                        pd.DataFrame(val_loss, columns=['val_loss'])], axis=1)

    result.to_csv('E:results/maf_cnn{}.csv'.format(k + 1))


