import torch
import torch.nn as nn
import hdf5storage
import numpy as np
from scipy.io import loadmat


def tuple_transpose(tuple_data):
    transposed_arrays = []
    for arr in tuple_data:
        transposed_arrays.append(np.transpose(arr))
    return transposed_arrays


def tuple_transpose_squeeze(tuple_data):
    transposed_arrays = []
    for arr in tuple_data:
        transposed_arrays.append(np.squeeze(arr.T))
    return transposed_arrays


def generator_data2(choice, choice2):    # 2025.1.7 数据集是9个特征的数据
    path = ''
    if choice == 'all':
        path = '_all'
    elif choice == 'GZH':
        path = '_GZH.mat'
    elif choice == 'LS':
        path = '_LS.mat'
    elif choice == 'LXY':
        path = '_LXY.mat'
    elif choice == 'TZH':
        path = '_TZH.mat'
    elif choice == 'YYF':
        path = '_YYF.mat'
    filename1 = "{}{}".format("./data/Train_X", path)
    filename2 = "{}{}".format("./data/Train_Y", path)

    X_Train = hdf5storage.loadmat(filename1)['Train_X'][:, :].flatten()
    X_Train = tuple_transpose(X_Train)

    Y_Train = hdf5storage.loadmat(filename2)['Train_Y'][:, :].flatten()
    Y_Train = tuple_transpose_squeeze(Y_Train)

    # normalization
    muX, sigmaX = normalization_x(X_Train)
    muY, sigmaY = normalization_y(Y_Train)
    scaler = np.array([muY, sigmaY])

    x_train = [(x - muX) / sigmaX for x in X_Train]
    y_train = [(y - muY) / sigmaY for y in Y_Train]
    return x_train, y_train, scaler


def normalization_x(data):
    train_data = np.vstack(data)
    muX = np.mean(train_data, axis=0)
    sigmaX = np.std(train_data, axis=0)
    return muX, sigmaX


def normalization_y(data):
    train_data = np.concatenate(data)
    muY = np.mean(train_data, axis=0)
    sigmaY = np.std(train_data, axis=0)
    return muY, sigmaY


if __name__ == '__main__':
    X, Y, scaler = generator_data2('all', 'train')
    muX, sigmaX = normalization_x(X)
    print(muX, sigmaX)