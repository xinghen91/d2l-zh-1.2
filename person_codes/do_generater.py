

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

g2 = generator_func2()
print(next(g2))
g2.send('content 111')
print(next(g2))