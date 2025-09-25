#装饰器类，统计函数执行次数
from functools import wraps
class CallingParameter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0
        self.velocity = []
        self.temperature = []
        self.a1 = []
        self.a2 = []

    def __call__ (self, *args, **kwargs):
        self.count += 1
        self.velocity = self.velocity + [kwargs['velocity']]
        self.temperature = self.temperature + [kwargs['temperature']]
        self.a1 = self.a1 + [kwargs['a1']]
        self.a2 = self.a2 + [kwargs['a2']]
        return self.func(*args, **kwargs)

class CallingCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __get__(self, instance, owner):
        def wrapper(*args, **kwargs):
            self.count += 1
            wrapper.count = self.count   # ✅ 把 count 也挂到 wrapper 上
            return self.func(instance, *args, **kwargs)
        wrapper.count = self.count
        return wrapper
