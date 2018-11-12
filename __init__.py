# -*- coding:utf-8 -*-
from LSTM_model import lstm_model
from util import read_data
import numpy as np

x_train = np.load("data/train_data/x_train.npy")
y_train = np.load("data/train_data/y_train.npy")
x_test = np.load("data/test_data/x_test.npy")
y_test = np.load("data/test_data/y_test.npy")

index = 1001
layer_num = 1 
cell_num = 300 
hidden_num = 50
nb_epoch = 10000
dropout = 0.1
lr = 0.001
batch_size = 25

lstm_model(index, x_train, y_train, x_test, y_test, layer_num, cell_num, hidden_num, nb_epoch, dropout, lr, batch_size)
