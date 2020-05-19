import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    '''
    block to create net
    '''
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.ouput = nn.Dense(10)


    def forward(self, x):
        return self.ouput(self.hidden(x))
        pass

class MySequential(nn.Block):
    '''
    persnoal add forward function block to create net
    '''
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block


    def forward(self, x):
        for block in self._children.values():
            x = block(x)

        return x

class FancyMlp(nn.Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20,20))
        )
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        print(x)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /= 2

        while x.norm().asscalar() < 0.8:
            x *= 10

        return x.sum()

def call_fancy_mlp():
    x = nd.random.uniform(shape=(2,20))
    net = FancyMlp()
    net.initialize()
    net(x)




            
def call_mysequential():
    x = nd.random.uniform()
    net = MySequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    net.initialize()
    net(x)





def call_mlp():

    x = nd.random.uniform()
    net = MLP()
    net.initialize()
    net(x)
    print(net.collect_params())


if __name__ == '__main__':
    # call_mysequential()
    call_fancy_mlp()