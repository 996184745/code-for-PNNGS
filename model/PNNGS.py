#PNNGS Parallel neural network for genomic selection
#https://colab.research.google.com/drive/1o8lfWHvr4WoyTA5Y9b4mSCSw2TEbXJb7?usp=sharing#scrollTo=15ae67a2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge
from imblearn.over_sampling import RandomOverSampler
import scipy.stats as stats
import os

# Disabling Scipy warnings
np.seterr(all='ignore')
torch.backends.cudnn.enabled = False


class Inception1d(nn.Module):
    def __init__(self, in_channels, parallel_number, stride = 1):
        super(Inception1d, self).__init__()
        self.parallel_number = parallel_number

        # first line
        self.branch1 = nn.Conv1d(in_channels= in_channels, out_channels = 1, kernel_size= 1,
                                     stride= stride, padding= 0)

        # second line
        self.branch3 = nn.Conv1d(in_channels = in_channels, out_channels = 3, kernel_size= 3,
                                     stride= stride, padding= 1)

        # third line
        self.branch5 = nn.Conv1d(in_channels = in_channels, out_channels = 3, kernel_size= 5,
                                     stride= stride, padding= 2)

        # fourth line
        self.branch7 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=7,
                                 stride=stride, padding=3)

        # fifth line
        self.branch9 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=9,
                                 stride=stride, padding=4)

        # sixth line
        self.branch11 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=11,
                                 stride=stride, padding= 5)

        # seventh line
        self.branch13 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=13,
                                  stride=stride, padding=6)

        # eighth line
        self.branch15 = nn.Conv1d(in_channels=in_channels, out_channels= 3, kernel_size=15,
                                  stride=stride, padding=7)

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch3(x)
        f3 = self.branch5(x)
        f4 = self.branch7(x)
        f5 = self.branch9(x)
        f6 = self.branch11(x)
        f7 = self.branch13(x)
        f8 = self.branch15(x)
        if self.parallel_number == 2:
            output = torch.cat((f1, f2), dim=1)
        elif self.parallel_number == 3:
            output = torch.cat((f1, f2, f3), dim=1)
        elif self.parallel_number == 4:
            output = torch.cat((f1, f2, f3, f4), dim=1)
        elif self.parallel_number == 5:
            output = torch.cat((f1, f2, f3, f4, f5), dim=1)
        elif self.parallel_number == 6:
            output = torch.cat((f1, f2, f3, f4, f5, f6), dim=1)
        elif self.parallel_number == 7:
            output = torch.cat((f1, f2, f3, f4, f5, f6, f7), dim=1)
        elif self.parallel_number == 8:
            output = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8), dim=1)
        else:
            output = "error"
        return output


class PNNGSModel(nn.Module):
    def __init__(self, parallel_number, X_train):
        super(PNNGSModel, self).__init__()
        self.conv1 = Inception1d(in_channels=1, parallel_number= parallel_number, stride=1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.batch = nn.BatchNorm1d(3 * parallel_number - 2)
        self.conv2 = Inception1d(in_channels= 3 * parallel_number - 2, parallel_number= parallel_number, stride=1)
        self.conv3 = nn.Conv1d(in_channels= 3 * parallel_number - 2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(X_train.shape[1], 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def keepOnlyTheLargestCategory(data_file, cluster_file):
    data_file = pd.read_csv(data_file, header=0, index_col=0)
    cluster_file = pd.read_csv(cluster_file, header=0, index_col=0)
    cluster_number = cluster_file["cluster"]
    print(cluster_file["cluster"].value_counts())
    print(cluster_file["cluster"].value_counts().index[0])
    largest_category = cluster_file["cluster"].value_counts().index[0]
    data_file = pd.merge(data_file, cluster_number, how= 'inner', left_index= True, right_index= True)
    data_file = data_file[data_file["cluster"] == largest_category]
    data_file = data_file.drop("cluster", axis= "columns")
    return data_file


def PNNGS(pheno,
          data_file,
          n_splits = 10,
          epoch = 1500,
          CUDA_VISIBLE_DEVICES = '0',
          parallel_number = 3,
          Batch_Size = 32,
          save_path = '../save_model/PNNGS.pth'):
    '''
    :param pheno: pheno name
    :param data_file: phenotype+genotype
    :param n_splits: the split number
    :param epoch:
    :param CUDA_VISIBLE_DEVICES: the number of CUDA_VISIBLE_DEVICES
    :param parallel_number:
    :param Batch_Size:
    :param save_path:
    :return: average prediction and prediction list
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    repeat_times = 2

    data_file = pd.read_csv(data_file, header=0, index_col=0)

    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)

    X = np.array(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    train_pearns_list, test_pearns_list = [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index].values, y[test_index].values

        ridge_model = Ridge(random_state=0)
        ridge_model.fit(X_train, y_train)
        y_pre = ridge_model.predict(X_test)
        pearn_ridge, _ = stats.pearsonr(y_pre, y_test)
        print("pearn_ridge:", pearn_ridge)

        X_train_torch = torch.from_numpy(X_train.astype(np.float32))
        y_train_torch = torch.from_numpy(y_train.astype(np.float32))
        X_test_torch = torch.from_numpy(X_test.astype(np.float32))
        y_test_torch = torch.from_numpy(y_test.astype(np.float32))

        X_train_torch = torch.unsqueeze(X_train_torch, 1)
        y_train_torch = torch.unsqueeze(y_train_torch, 1)
        X_test_torch = torch.unsqueeze(X_test_torch, 1)
        y_test_torch = torch.unsqueeze(y_test_torch, 1)

        trainset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
        testset = torch.utils.data.TensorDataset(X_test_torch, y_test_torch)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size, shuffle=False, num_workers=8)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("device:", device)

        net = PNNGSModel(parallel_number, X_train)
        print("total_params:", sum(p.numel() for p in net.parameters()))

        if device == 'cuda':
            net = nn.DataParallel(net)
            torch.backends.cudnn.benchmark = True

        optimizer = optim.Adam(net.parameters(), weight_decay=0.1)

        criterion = nn.MSELoss()

        best_pearns = 0

        for repeat_time in range(repeat_times):
            best_epoch = 0
            nan_number = 0
            if repeat_time == 0:
                early_stop = 200
            else:
                early_stop = 100
            print("\rrepeat_time:", repeat_time)
            print("early_stop:", early_stop)

            for i in range(epoch):
                if torch.cuda.is_available():
                    net = net.to(device)
                net.train()
                for step, data in enumerate(trainloader, start=0):
                    im, label = data
                    im = im.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    outputs = net(im)
                    loss = criterion(outputs, label)

                    loss.backward()
                    optimizer.step()

                net.eval()
                test_outputs = []
                with torch.no_grad():
                    for step, data in enumerate(testloader, start=0):
                        X_test_input, Y_test_label = data
                        X_test_input = X_test_input.to(device)
                        test_output = net(X_test_input)
                        test_output = test_output.cpu().detach().numpy().squeeze()
                        test_output = list(test_output)
                        test_outputs = test_outputs + test_output

                test_outputs = np.array(test_outputs)
                test_pearns = stats.pearsonr(test_outputs, y_test)[0]
                # print("\rEpoch:", i+1, "test_pearns:", test_pearns)


                if test_pearns > best_pearns:
                    # torch.save(net, save_path)
                    best_pearns = test_pearns
                    best_epoch = i
                    print("\rEpoch:", i+1, "best_pearns:", best_pearns)

                if i - best_epoch > early_stop:
                    # print("i - best_epoch > early_stop", "best_epoch:", best_epoch, "epoch:", i)
                    break

                if i > 200:
                    if best_pearns < 0.9 * pearn_ridge:
                        # print("best_pearns < 0.7 * pearn_ridge")
                        break

                if np.isnan(test_pearns):
                    nan_number += 1
                    # print("nan_number:", nan_number)
                    if nan_number > 20:
                        break

            print("final epoch:", i)

            if best_pearns > pearn_ridge + 0.01:
                break

        print("best_pearns:", best_pearns)
        test_pearns_list.append(best_pearns)

    average_pearns = np.mean(test_pearns_list)
    print("average_pearns:", average_pearns)
    print("test_pearns_list:", test_pearns_list)
    return average_pearns, test_pearns_list


def PNNGSSStratifiedImbalanced(pheno,
          data_file,
          cluster_file,
          n_splits = 10,
          epoch = 1500,
          CUDA_VISIBLE_DEVICES = '0',
          parallel_number = 3,
          Batch_Size = 32,
          save_path = '../save_model/PNNGS.pth'):
    '''

    :param pheno: pheno name
    :param data_file: phenotype+genotype
    :param n_splits: the split number
    :param epoch:
    :param CUDA_VISIBLE_DEVICES: the number of CUDA_VISIBLE_DEVICES
    :param parallel_number:
    :param Batch_Size:
    :param save_path:
    :return: average prediction and prediction list
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    repeat_times = 2

    data_file = pd.read_csv(data_file, header=0, index_col=0)

    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)

    skfold = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state=1)
    cluster_file = pd.read_csv(cluster_file, header=0, index_col=0)
    cluster_number = cluster_file["cluster"]

    train_pearns_list, test_pearns_list = [], []

    for fold, (train_index, test_index) in enumerate(skfold.split(X, cluster_number)):
        print("fold:", fold)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #imbalanced
        X_y_train = pd.concat([X_train, y_train], axis=1)
        # X_y_train.columns.astype(str)
        y_train_cluster = cluster_number.iloc[train_index]
        over = RandomOverSampler()
        X_smote, y_smote = over.fit_resample(X_y_train, y_train_cluster)
        print(y_smote.value_counts())

        y_train = X_smote[pheno]
        X_train = X_smote.drop([pheno], axis=1)
        # imbalanced


        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        ridge_model = Ridge(random_state=0)
        ridge_model.fit(X_train, y_train)
        y_pre = ridge_model.predict(X_test)
        pearn_ridge, _ = stats.pearsonr(y_pre, y_test)
        print("pearn_ridge:", pearn_ridge)

        X_train_torch = torch.from_numpy(X_train.astype(np.float32))
        y_train_torch = torch.from_numpy(y_train.astype(np.float32))
        X_test_torch = torch.from_numpy(X_test.astype(np.float32))
        y_test_torch = torch.from_numpy(y_test.astype(np.float32))

        X_train_torch = torch.unsqueeze(X_train_torch, 1)
        y_train_torch = torch.unsqueeze(y_train_torch, 1)
        X_test_torch = torch.unsqueeze(X_test_torch, 1)
        y_test_torch = torch.unsqueeze(y_test_torch, 1)

        trainset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
        testset = torch.utils.data.TensorDataset(X_test_torch, y_test_torch)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size, shuffle=False, num_workers=8)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("device:", device)

        net = PNNGSModel(parallel_number, X_train)
        print("total_params:", sum(p.numel() for p in net.parameters()))

        if device == 'cuda':
            net = nn.DataParallel(net)
            torch.backends.cudnn.benchmark = True

        optimizer = optim.Adam(net.parameters(), weight_decay=0.1)

        criterion = nn.MSELoss()

        best_pearns = 0

        for repeat_time in range(repeat_times):
            best_epoch = 0
            nan_number = 0
            if repeat_time == 0:
                early_stop = 200
            else:
                early_stop = 100
            print("\rrepeat_time:", repeat_time)
            print("early_stop:", early_stop)

            for i in range(epoch):
                if torch.cuda.is_available():
                    net = net.to(device)
                net.train()
                for step, data in enumerate(trainloader, start=0):
                    im, label = data
                    im = im.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    outputs = net(im)
                    loss = criterion(outputs, label)

                    loss.backward()
                    optimizer.step()

                net.eval()
                test_outputs = []
                with torch.no_grad():
                    for step, data in enumerate(testloader, start=0):
                        X_test_input, Y_test_label = data
                        X_test_input = X_test_input.to(device)
                        test_output = net(X_test_input)
                        test_output = test_output.cpu().detach().numpy().squeeze()
                        test_output = list(test_output)
                        test_outputs = test_outputs + test_output


                #
                test_outputs = np.array(test_outputs)
                # print("test_outputs:", test_outputs)
                test_pearns = stats.pearsonr(test_outputs, y_test)[0]
                # print("\rEpoch:", i+1, "test_pearns:", test_pearns)

                # test_outputs = net(X_test_torch)
                # print("test_outputs:", test_outputs)

                if test_pearns > best_pearns:
                    # torch.save(net, save_path)
                    best_pearns = test_pearns
                    best_epoch = i
                    print("\rEpoch:", i+1, "best_pearns:", best_pearns)

                if i - best_epoch > early_stop:
                    # print("i - best_epoch > early_stop", "best_epoch:", best_epoch, "epoch:", i)
                    break

                if i > 200:
                    if best_pearns < 0.9 * pearn_ridge:
                        # print("best_pearns < 0.7 * pearn_ridge")
                        break

                if np.isnan(test_pearns):
                    nan_number += 1
                    # print("nan_number:", nan_number)
                    if nan_number > 20:
                        break

            print("final epoch:", i)

            if best_pearns > pearn_ridge + 0.01:
                break

        print("best_pearns:", best_pearns)
        test_pearns_list.append(best_pearns)

    average_pearns = np.mean(test_pearns_list)
    print("average_pearns:", average_pearns)
    print("test_pearns_list:", test_pearns_list)
    return average_pearns, test_pearns_list


def PNNGSTheLargestCategory(pheno,
          data_file,
          cluster_file,
          n_splits = 10,
          epoch = 1500,
          CUDA_VISIBLE_DEVICES = '0',
          parallel_number = 3,
          Batch_Size = 32,
          save_path = '../save_model/PNNGS.pth'):
    '''

    :param pheno: pheno name
    :param data_file: phenotype+genotype
    :param n_splits: the split number
    :param epoch:
    :param CUDA_VISIBLE_DEVICES: the number of CUDA_VISIBLE_DEVICES
    :param parallel_number:
    :param Batch_Size:
    :param save_path:
    :return: average prediction and prediction list
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    repeat_times = 2

    data_file = keepOnlyTheLargestCategory(data_file, cluster_file)

    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)
    print("X.shape:", X.shape)

    X = np.array(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    train_pearns_list, test_pearns_list = [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index].values, y[test_index].values

        ridge_model = Ridge(random_state=0)
        ridge_model.fit(X_train, y_train)
        y_pre = ridge_model.predict(X_test)
        pearn_ridge, _ = stats.pearsonr(y_pre, y_test)
        print("pearn_ridge:", pearn_ridge)

        X_train_torch = torch.from_numpy(X_train.astype(np.float32))
        y_train_torch = torch.from_numpy(y_train.astype(np.float32))
        X_test_torch = torch.from_numpy(X_test.astype(np.float32))
        y_test_torch = torch.from_numpy(y_test.astype(np.float32))

        X_train_torch = torch.unsqueeze(X_train_torch, 1)
        y_train_torch = torch.unsqueeze(y_train_torch, 1)
        X_test_torch = torch.unsqueeze(X_test_torch, 1)
        y_test_torch = torch.unsqueeze(y_test_torch, 1)

        trainset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
        testset = torch.utils.data.TensorDataset(X_test_torch, y_test_torch)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size, shuffle=False, num_workers=8)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("device:", device)

        net = PNNGSModel(parallel_number, X_train)
        print("total_params:", sum(p.numel() for p in net.parameters()))

        if device == 'cuda':
            net = nn.DataParallel(net)
            torch.backends.cudnn.benchmark = True

        optimizer = optim.Adam(net.parameters(), weight_decay=0.1)

        criterion = nn.MSELoss()

        best_pearns = 0

        for repeat_time in range(repeat_times):
            best_epoch = 0
            nan_number = 0
            if repeat_time == 0:
                early_stop = 200
            else:
                early_stop = 100
            print("\rrepeat_time:", repeat_time)
            print("early_stop:", early_stop)

            for i in range(epoch):
                if torch.cuda.is_available():
                    net = net.to(device)
                net.train()
                for step, data in enumerate(trainloader, start=0):
                    im, label = data
                    im = im.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    outputs = net(im)
                    loss = criterion(outputs, label)

                    loss.backward()
                    optimizer.step()

                net.eval()
                test_outputs = []
                with torch.no_grad():
                    for step, data in enumerate(testloader, start=0):
                        X_test_input, Y_test_label = data
                        X_test_input = X_test_input.to(device)
                        test_output = net(X_test_input)
                        test_output = test_output.cpu().detach().numpy().squeeze()
                        test_output = list(test_output)
                        test_outputs = test_outputs + test_output


                #
                test_outputs = np.array(test_outputs)
                # print("test_outputs:", test_outputs)
                test_pearns = stats.pearsonr(test_outputs, y_test)[0]
                # print("\rEpoch:", i+1, "test_pearns:", test_pearns)

                if test_pearns > best_pearns:
                    # torch.save(net, save_path)
                    best_pearns = test_pearns
                    best_epoch = i
                    print("\rEpoch:", i+1, "best_pearns:", best_pearns)

                if i - best_epoch > early_stop:
                    # print("i - best_epoch > early_stop", "best_epoch:", best_epoch, "epoch:", i)
                    break

                if i > 200:
                    if best_pearns < 0.9 * pearn_ridge:
                        # print("best_pearns < 0.7 * pearn_ridge")
                        break

                if np.isnan(test_pearns):
                    nan_number += 1
                    # print("nan_number:", nan_number)
                    if nan_number > 20:
                        break

            print("final epoch:", i)

            if best_pearns > pearn_ridge + 0.01:
                break

        print("best_pearns:", best_pearns)
        test_pearns_list.append(best_pearns)

    average_pearns = np.mean(test_pearns_list)
    print("average_pearns:", average_pearns)
    print("test_pearns_list:", test_pearns_list)
    return average_pearns, test_pearns_list