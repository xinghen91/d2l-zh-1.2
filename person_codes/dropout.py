import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn


def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob

    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1)
    return mask * X / keep_prob


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
drop_prob1, drop_prob2 = 0.2, 0.5
for param in params:
    param.attach_grad()


def net(X):
    pro1, pro2 = 0.2, 0.5
    X = X.reshape(-1, num_inputs)
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():
        dropout(H1, drop_prob1)
    H2 = nd.dot(H1, W2) + b2
    if autograd.is_training():
        dropout(H2, drop_prob2)
    return nd.dot(H2, W3) + b3


num_epochs, lr, batch_size = 25, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dropout(drop_prob1),
        nn.Dense(256, activation='relu'),
        nn.Dropout(drop_prob2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,None, None, trainer)
