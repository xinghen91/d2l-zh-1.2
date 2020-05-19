import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6,], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
ploy_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
lables = true_w[0] * ploy_features[:,0] + true_w[1] * ploy_features[:,1] +true_w[2] * ploy_features[:,2] + true_b
lables += nd.random.normal(scale=0.1, shape=lables.shape)
print(features[:2])
print(ploy_features[:2])
print(lables[:2])

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_vals)
    d2l.plt.ylabel(y_vals)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)



