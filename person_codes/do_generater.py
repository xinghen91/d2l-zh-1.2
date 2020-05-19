

def generator_fun1():
    a = 1
    print('-'*20 + 'def arg a ' + '-'*20)

    yield a

    b=2
    print('-'*20 + 'def arg a ' + '-'*20)
    yield b


# g1 = generator_fun1()
#
# print(next(g1))
# print(next(g1))
# print(next(g1))


def generator_func2():
    print(123)
    content = yield 1
    print(content + "============")

    print(123)

    yield 2


# g2 = generator_func2()
# print(next(g2))
# g2.send('content 111')
# print(next(g2))


def gen_fun3 ():

    for c in "AB":
        yield c

    for i in range(3):
        yield  i
def gen_fun3_2():
    yield from "AB"
    yield from range(3)



# print(list(gen_fun3()))
# print(list(gen_fun3_2()))


def list_gen():
    egg_list = ['eggs %s' %i for i in range(10)]
    print(egg_list)

    mj = ('eggs from mj %s' %i for i in range(10))
    print(mj)
    print(mj.__next__())
    print(next(mj))
    print(next(mj))


# list_gen()
# print(sum(x ** 2 for x in range(100000000)))
# print(sum([x ** 2 for x in range(100000000)]))

def demo():
    for i in range(4):
        yield i

# g=demo()
#
# g1=(i for i in g)
# g2=(i for i in demo())
#
# print(list(g1))
# print(list(g2))

def test():
    for i in range(4):
        yield i

def add(n, i):
    print('n:' + str(n) + ',i:' +str(i))
    return n+i

g = test()

for n in range(10):
    print(n)
    g = (add(n, i) for i in g)
    # print(list(g))

print(list(g))