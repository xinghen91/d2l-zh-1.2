def wrapper1(func_name):
    print('function :', func_name)

    def wrapper(argv):
        print('function : ', func_name)
        if login(argv):
            return func_name(argv)
        else:
            return 'error'

    return wrapper


def login(user):
    if user == "A":
        return True


def wrapper2(func_name):
    def func():
        print('w2, before func')
        func_name()
        print('w2, after func')

    return func


def wrapper3(func_name):
    def func():
        print('w3, before func')
        func_name()
        print('w3, after func')

    return func


@wrapper3
@wrapper2
# @wrapper1()
def home():
    # result = login('A')
    # if result:
    print("there is home page!")
    # print('hello ' )


def Before(request):
    print ('before')

def After(request):
    print ('after')

def Filter(before_func,after_func):
    def outer(main_func):
        def wrapper(request):
            before_result = before_func(request)
            if(before_result != None):
                return before_result
            main_result = main_func(request)
            if(main_result != None):
                return main_result
            after_result = after_func(request)
            if(after_result != None):
                return after_result
        return wrapper
    return outer

@Filter(Before, After)
def Index(request):
    print ('index')

#
#
if __name__ == '__main__':
    Index('example')

#     # home = wrapper(home)
#     # home()
#     # home = wrapper1(home)
#     # home('A')
#     index()
