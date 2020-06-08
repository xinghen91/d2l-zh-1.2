import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

def dropout (X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob

    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1)