import pandas as pd
import numpy as np
import d2lzh as d2l
from mxnet import autograd, nd, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn

test_data = pd.read_csv('kaggle_house_pred_test.csv')
train_data = pd.read_csv('kaggle_house_pred_train.csv')

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features)
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print(numeric_features)

all_features[numeric_features] = all_features[numeric_features].apply(lambda x: x - x.mean() / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# print(all_features[numeric_features])
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
n_train = all_features.shape[0]
train_features = nd.array(all_features[:train_data].valus)
test_features = nd.array(all_features[:train_data].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))
loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net


def log_rmse(net, features, labels):
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    # assert isinstance(clipped_preds, nd.ndarray)
    rmse = nd.sqrt(2*(loss(clipped_preds.log(), labels.log()).mean()))
    return rmse.asscalar()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learnning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True
    )

