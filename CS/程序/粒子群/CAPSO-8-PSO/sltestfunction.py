import numpy as np
import random

'''unimodal function单峰函数f1-f8'''


def f1(x):
    return np.sum(x**2)


def f2(x):
    dim = x.shape[0]
    res = 0
    for i in range(dim):
        res = res + np.sum(x[:i+1])**2
    return res


def f3(x):
    x = np.abs(x)
    return np.max(x)

def f4(x):
    x = np.abs(x)
    return np.sum(x) + np.prod(x)

def f5(x):
    dim = x.shape[0]
    x = np.floor(x + 0.5)
    return np.sum(x[:dim] ** 2)


def f6(x):
    ss = 0
    for i in range(x.shape[0]):
        ss += np.power(abs(x[i]), i+2)
    return ss


def f7(x):
    return x[0]*x[0] + 10**6 * (np.sum(x**2) - x[0]*x[0])


def f8(x):
    return 10**6 * (x[0]*x[0]) + np.sum(x**2) - x[0]**2

def f9(x):
    return np.sum(x**2) + np.power(np.sum(0.5*x), 2) + np.power(np.sum(0.5*x), 4)


"""multimodal function多峰函数f9-f16"""

def f10(x):
    dim = x.shape[0]
    res = 0
    for i in range(dim-1):
        res = res + 100 * (np.power(x[i+1] - x[i]**2, 2)) + (x[i] - 1)**2

    return res


def f11(x):
    dim = x.shape[0]
    res = 0
    for i in range(dim - 1):
        res = res + (i+1)*(x[i]**4)
    return res + np.random.random()

def f12(x):
    return np.sum(abs(x*(np.sin(x)) + 0.1 * x))

def f13(x):  # 测试无误
    return np.sum(-x*np.sin(np.sqrt(np.abs(x))))


def f14(x):
    return np.sum(x**2 - 10 * np.cos(2*np.pi*x) + 10)


def f15(x):  # 测试无误
    dim = x.shape[0]
    return -20 * np.exp(-0.2 * np.sqrt((1/dim) * np.sum(x**2))) + 20 - np.exp((1/dim)*np.sum(np.cos(2 * np.pi * x))) + np.e


def f16(x):  # 测试无误
    dim = x.shape[0]
    res = 1
    for i in range(dim):
        res = res * np.cos((x[i]/np.sqrt(i+1)))
    return (1/4000) * np.sum(x**2) - res + 1


def f17(x):  # 测试无误
    dim = x.shape[0]
    y = 1 + 0.25 * (x + 1)
    res1 = 10 * (np.sin(np.pi * y[0]) ** 2)
    res2 = 0
    res3 = (y[dim - 1] - 1) ** 2
    res4 = 0
    a = 10
    for ii in range(dim):
        if x[ii] > a:
            u = 100 * ((x[ii] - a) ** 4)
        elif x[ii] < -a:
            u = 100 * ((-x[ii] - a) ** 4)
        else:
            u = 0
        res4 = res4 + u
    for i in range(dim - 1):
        res2 += ((y[i] - 1) ** 2) * (1 + 10 * (np.sin(np.pi * y[i + 1]) ** 2))
    return (np.pi / dim) * (res1 + res2 + res3) + res4


def f18(x):  # 测试无误
    dim = x.shape[0]
    res1 = 0
    res2 = 0
    res4 = 0
    a = 5
    for i in range(dim - 1):
        res1 = res1 + (x[i] - 1)**2 * (1 + np.power(np.sin(3 * np.pi * x[i+1]), 2)) +\
               np.power((x[dim-1]-1), 2) * (1 + np.sin(2*np.pi*x[dim-1])**2)
    for ii in range(dim):
        if x[ii] > a:
            u = 100 * ((x[ii] - a) ** 4)
        elif x[ii] < -a:
            u = 100 * ((-x[ii] - a) ** 4)
        else:
            u = 0
        res4 = res4 + u
    return 0.1 * (np.power(np.sin(np.pi * 3 * x[0]), 2) + res1) + res2



f_list1 = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13,f14,f15,f16,f17,f18]  # 单峰函数f1-f7, 多峰函数f8-f13



# data = np.array(
#     [0.44992408,  0.47258551,  0.22088175,  0.72238484,  0.63467501])
# # data1 = np.array([0.44992408,  0.47258551,  0.22088175,  0.72238484,  0.63467501, 0.54463335,  0.31463825,  0.61308303,  0.98978476,  0.64978096])
# # data2 = np.array([0.44992408,  0.47258551,  0.22088175,  0.72238484,  0.63467501, 0.54463335,  0.31463825,  0.61308303,  0.98978476,  0.64978096])
# # data3 = np.array([0.44992408,  0.47258551])
# # data4 = np.array([0.44992408,  0.47258551,  0.22088175,  0.72238484])
# # data5 = np.array([0.44992408,  0.47258551,  0.22088175])
# # data6 = np.array([0.44992408,  0.47258551,  0.22088175,  0.72238484,  0.63467501, 0.54463335])
# for f in f_list1:
#     print(f.__name__, f(data))
#     print('------------------')