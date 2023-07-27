def decorator(func):
    def wrapper(self, *args, **kwargs):
        print('Before calling the function')
        print('Access instance attribute: ', self.attribute)
        result = func(self, *args, **kwargs)
        print('After calling the function')
        return result
    return wrapper


class MyClass:
    def __init__(self):
        self.attribute = "Hello"

    @decorator
    def say_hello(self):
        print('Inside the method')


obj = MyClass()
obj.say_hello()
