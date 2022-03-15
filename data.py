# -*- coding:utf-8 -*-
"""
作者：ooswald216
日期：2022年03月14
数据载入模块
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

train = pd.read_csv(r"data/train.csv",dtype = np.float32)
val = pd.read_csv("data/test.csv",dtype = np.float32)
#像素值范围0-255，/255归一化
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255
features_val = val.loc[:,val.columns != "label"].values/255

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_numpy, test_size = 0.2, random_state = 42)

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
featuresVal = torch.from_numpy(features_val).type(torch.LongTensor)

batch_size = 100
n_iters = 2500
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)
# 对给定的tensor数据(样本和标签)，将它们包装成dataset 注意，如果是numpy的array，或者Pandas的DataFrame需要先转换成Tensor
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)
val = torch.utils.data.TensorDataset(featuresVal)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
val_loader = DataLoader(val,batch_size = batch_size, shuffle = False)
# plt.imshow(features_numpy[10].reshape(28,28))
# plt.axis("off")
# plt.title(str(targets_numpy[10]))
# plt.savefig('graph.png')
# plt.show()