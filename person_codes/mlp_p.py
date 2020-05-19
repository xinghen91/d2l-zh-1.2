import d2lzh as d2l
from mxnet import nd
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import loss as gloss
batch_size = 256
mnist = d2l.load_data_fashion_mnist(batch_size)
train_iter, test_iter = mnist


num_input, num_out, num_hidden = 784, 10, 256

w1 = nd.random.normal(loc=0, scale=0.01, shape=(num_input, num_hidden))
b1 = nd.zeros(shape=(num_hidden,))
w2 = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_out))
b2 = nd.zeros(shape=(num_out,))
params = [w1, b1, w2, b2]
for param in params:
    param.attach_grad()


def relu(x):
    return nd.maximum(x, 0)


def net(x):
    x = x.reshape(-1, num_input)
    H = relu(nd.dot(x, w1) + b1)
    return nd.dot(H, w2) + b2


loss = gloss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 100, 0.5


def train_ch3(net, train_iter, test_iter, loss, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

        for x, y in train_iter:
            with mx.autograd.record():
                y_hat = net(x)
                l = loss(y_hat, y).sum()
            l.backward()

            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer(params, lr, batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
        train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1 , train_l_sum, train_acc_sum/n, test_acc))

train_ch3(net, train_iter, test_iter, loss, batch_size, params, lr)