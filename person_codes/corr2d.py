
from mxnet import autograd, nd
from mxnet.gluon import nn


def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shae[0]):
        for j in range(Y.shapq[1]):
            Y[i, j] = (X[i:i + h][j:j + w] * K).sum()
    return Y


x = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[1, 2], [3, 4]])
corr2d(x, K)
