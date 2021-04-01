import numpy as np
import random

'''unimodal function单峰函数f1-f8'''

def f1(x):
    ss = 0
    for i in range(x.shape[0]):
        ss += np.power(abs(x[i]), i+2)
    return ss


def f2(x):
    return x[0]*x[0] + 10**6 * (np.sum(x**2) - x[0]*x[0])


def f3(x):
    return 10**6 * (x[0]*x[0]) + np.sum(x**2) - x[0]**2

def f4(x):
    return np.sum(x**2) + np.power(np.sum(0.5*x), 2) + np.power(np.sum(0.5*x), 4)


"""multimodal function多峰函数f9-f16"""

def f5(x):
    return np.sum(abs(x*(np.sin(x)) + 0.1 * x))


f_list1 = [f1, f2, f3, f4]  # 单峰补充
f_list2 = [f5]  # 多峰补充


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