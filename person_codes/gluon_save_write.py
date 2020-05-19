from mxnet import nd
from mxnet.gluon import  nn

x = nd.ones(shape=(10,2))
nd.save('x', x)
x2 = nd.load('x')

y = nd.zeros(2)

nd.save('xy', [x,y])
xy = nd.load('xy')


class MLP(nn.Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
y = net(x)

file_name = 'mlp.paramater'
net.save_parameters(file_name)

net2 = MLP()
net2.load_parameters(file_name)
y2 = net2(x)
print(y == y2)