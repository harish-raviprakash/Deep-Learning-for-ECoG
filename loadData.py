"""
TimeAR for ECoG signal classification
Code written by: Harish RaviPrakash
This file defines the function used to load the data
"""
import scipy.io as sio
import numpy as np
import scipy.stats as sms
from os.path import join


def load_data(fileName):
    """Input:
    file: Data with labels in 1st column
    """
    A = sio.loadmat(fileName)
    data_train = A['A']
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    return X_train, y_train


def read_data(timeFeats, root_path, psdPath, labelPath):
    """
    Read subject data and load into
    :param timeFeats: List of time domain features
    :param root_path: Path to the time domain features
    :param psdPath: Path to the PSD features
    :param labelPath: Path to channel labels
    :return: All features and labels
    """

    label = np.loadtxt(labelPath)
    A = []
    for i in range(len(timeFeats)):
        X, y = load_data(join(root_path, timeFeats[i] + '.mat'))
        A.append(X)
    blockSize = int(y.shape[0] / label.shape[0])
    X2, _ = load_data(psdPath)
    return A, X2, y, label, blockSize


def reshapeTimeFeats(A, y, indx, blockSize):
    B = []
    for i in range(len(A)):
        X, Y = reshapeForCV(A[i], y, indx, blockSize)
        B.append(X)
    return B, Y


def reshapeData(X):
    A = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return A


def earlyFusion(X, X_mean=None, X_std=None, training=True):
    """
    Combine the different time features into multichannel input
    :param X: List of different time features
    :param X_mean: Mean of training data
    :param X_std: standard deviation of training data
    :param training: boolean variable to determine whether training data or
                     testing data
    """
    A = []
    if training:
        X_mean = []
        X_std = []
        for i in range(len(X)):
            X1, X1_mean, X1_std = standardizeData(X[i], 0.0, 0.0, training)
            X_mean.append(X1_mean)
            X_std.append(X1_std)
            A.append(X1)
    else:
        for i in range(len(X)):
            X1, X1_mean, X1_std = standardizeData(X[i], X_mean[i], X_std[i], training)
            A.append(X1)
    B = np.stack(A, axis=2)
    return B, X_mean, X_std


def standardizeData(X1, dataMean=0.0, dataStd=0.0, training=True):
    if training:
        dataMean = np.mean(X1, axis=0)
        dataStd = np.std(X1, axis=0)
    A = (X1 - dataMean) / (dataStd)
    return A, dataMean, dataStd


def reshapeForCV(X, y, indx, blockSize):
    A = []
    B = []
    for i in indx:
        for j in range(blockSize):
            A.append(X[i*blockSize+j, :])
            B.append(y[i*blockSize+j])
    A = np.array(A)
    B = np.array(B)
    return A, B


def reshapeAR(X, indx, blockSize):
    A = []
    for i in indx:
        for j in range(blockSize):
            A.append(X[i*blockSize+j, :])
    A = np.array(A)
    return A


def blockToChannel(D, blockSize):
    """
    Majority voting
    """
    Y = []
    for i in range(0, len(D), blockSize):
        c, d = sms.mode(np.round(D[i:i+blockSize]))
        Y.append(c[0])
    K = np.reshape(np.asarray(Y), len(Y))
    return K


def reshapeOutputs(y):
    D = np.argmax(y, axis=1)
    return D


def probBlockApproach(D, blockSize):
    Y = []
    for i in range(0, D.shape[0], blockSize):
        t = np.mean(D[i:i+blockSize, :], 0)
        Y.append(t)
    return np.array(Y)


def predictions(D):
    """
    Different weighting for classification
    """
    Y = np.zeros((D.shape[0],))
    for i in range(D.shape[0]):
        if D[i, 0] > 0.35:
            continue
        else:
            Y[i] = 1
    return Y


def predictions2(D):
    """
    Different weighting for classification
    """
    Y = np.zeros((D.shape[0],))
    for i in range(D.shape[0]):
        if D[i, 0] > 0.65:
            continue
        else:
            Y[i] = 1
    return Y
