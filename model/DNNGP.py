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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
import os


class DNNGPModel(nn.Module):
    def __init__(self, out_channels, X_train):
        super(DNNGPModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels= 1, out_channels= out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.batch = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels= out_channels, out_channels= out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels= out_channels, out_channels= 1, kernel_size=3, stride=1, padding=1)
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


def DNNGP(pheno,
          data_file,
          n_splits = 10,
          epoch = 1500,
          CUDA_VISIBLE_DEVICES = '0',
          out_channels = 10,
          Batch_Size = 32,
          save_path = '../save_model/DNNGP.pth'):
    '''

    :param pheno: pheno name
    :param data_file: phenotype+genotype
    :param n_splits: the split number
    :param epoch:
    :param CUDA_VISIBLE_DEVICES: the number of CUDA_VISIBLE_DEVICES
    :param out_channels: the channel number of CNN
    :param Batch_Size:
    :param save_path:
    :return: average prediction and prediction list
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    early_stop = 500
    data_file = pd.read_csv(data_file, header= 0, index_col= 0)

    y = data_file[pheno]
    X = data_file.drop([pheno], axis=1)

    X = np.array(X)

    kf = KFold(n_splits= n_splits, shuffle= True, random_state= 0)

    train_pearns_list, test_pearns_list = [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index].values, y[test_index].values

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


        trainloader = torch.utils.data.DataLoader(trainset, batch_size = Batch_Size, shuffle = True, num_workers = 8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size, shuffle=False, num_workers=8)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("device:",device)

        net = DNNGPModel(out_channels, X_train)
        print("total_params:", sum(p.numel() for p in net.parameters()))

        if device == 'cuda':
            net = nn.DataParallel(net)
            torch.backends.cudnn.benchmark = True



        optimizer = optim.Adam(net.parameters(), weight_decay= 0.1)

        criterion = nn.MSELoss()


        best_pearns = 0
        best_epoch = 0

        for i in range(epoch):
            train_loss = 0
            if torch.cuda.is_available():
                net = net.to(device)
            net.train()
            for step, data in enumerate(trainloader, start= 0):
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

                train_loss += loss.data

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
            test_pearns = stats.pearsonr(test_outputs, y_test)[0]
            # print("\rEpoch:", i + 1, "test_pearns:", test_pearns)


            # test_loss_list.append(test_loss.item())

            # end = time.time()
            # print('\rEpoch [{:>3d}/{:>3d} Train Loss:{:>.6f}  Train Pearns:{:>.6f} Test Loss:{:>.6f}  Test Pearns:{:>.6f} Learning Rate:{:>.6f}]'.format(
            #     i+1, epoch, train_loss, train_pearns, test_loss, test_pearns, lr
            # ), end= '')
            # print()
            # print("best_pearns:", best_pearns)
            # time_ = int(end - start)
            # h = time_ / 3600
            # m = time_ % 3600 / 60
            # s = time_ % 60
            # time_str = "\tTime %02d:%02d" % (m, s)
            #
            # print(time_str)

            if test_pearns > best_pearns:
                # torch.save(net, save_path)
                best_pearns = test_pearns
                best_epoch = i
                print("\rEpoch:", i+1, "best_pearns:", best_pearns)

            if i - best_epoch > early_stop:
                # print("i - best_epoch > early_stop", "best_epoch:", best_epoch, "epoch:", i)
                break


        print("best_pearns:", best_pearns)
        test_pearns_list.append(best_pearns)

    average_pearns = np.mean(test_pearns_list)
    print("average_pearns:", average_pearns)
    print("test_pearns_list:", test_pearns_list)
    return average_pearns, test_pearns_list