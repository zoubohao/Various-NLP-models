
### Test Class with __call__ method.
### In generally, it can add () after the object of this class
### and run the python code in its function.
class Test (object) :
    """
    ### Test Class with __call__ method.
    ### In generally, it can add () after the object of this class
    ### and run the python code in its function.
    """

    def __init__(self):
        pass

    def __call__(self, a , b):
        return a + b

test = Test()
print("###############")
print("Test call function.")
print(test(1.2,5.1))
print("###############")


### Test decorator
### The decorator in python is to dissolve the problem, code redundancy.
### The kernel code is the object which we want to decorate.
### By using the decorator, we can add additional feature around the kernel code
### but without writing another code once by once.

### In this form, The function TestDecorator can not use @ symbol to decorate
### the kernel code. Only can deliver one of object of kernel function to complete
### logical.
def TestDecorator(func,ifOutputName):
    """
    ### Test decorator
    ### The decorator in python is to dissolve the problem, code redundancy.
    ### The kernel code is the object which we want to decorate.
    ### By using the decorator, we can add additional feature around the kernel code
    ### but without writing another code once by once.

    ### In this form, The function TestDecorator can not use @ symbol to decorate
    ### the kernel code. Only can deliver one of object of kernel function to complete
    ### logical.
    :param func:
    :param ifOutputName:
    :return:
    """
    def wrapper(*args, **kwargs):
        print("This is an information before starting using function.")
        if ifOutputName:
            print("The parameter ifOutputName is True.")
            print("The name of this function is ", func.__name__)
        else:
            print("The parameter ifOutputName is False.")
        return func(*args, **kwargs)
    return wrapper

def TestFunc(a,b,ifAdd = True):
    if ifAdd:
        return a + b
    else:
        return a * b
test = TestDecorator(TestFunc,True)(1,2.0,False)
print(test)
print("#################")
### In this form, The TestDecorator1 only has one parameter.
### It can use @ symbol to substitute the assignment code:
### test = TestDecorator(TestFunc,True)(1,2.0,False)
def TestDecorator1(ifOutputName) :
    """
    ### In this form, The TestDecorator1 only has one parameter.
    ### It can use @ symbol to substitute the assignment code:
    ### test = TestDecorator(TestFunc,True)(1,2.0,False)
    :param ifOutputName:
    :return:
    """
    def decorator(func) :
        def wrapper(*args ,  **kwargs) :
            print("This is an information before starting using function.")
            if ifOutputName:
                print("The parameter ifOutputName is True.")
                print("The name of this function is ", func.__name__)
            else:
                print("The parameter ifOutputName is False.")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@TestDecorator1(False)
def TestFunc1(a,b,ifAdd = True):
    if ifAdd:
        return a + b
    else:
        return a * b
print(TestFunc1(1,2.0,True))



