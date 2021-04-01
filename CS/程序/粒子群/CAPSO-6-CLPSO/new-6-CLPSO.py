import numpy as np
import random
import resltestfunction
import matplotlib.pyplot as plt
import pandas as pd
import time


class CLPSO:
    def __init__(self, pn, dim, iter_max, xmax, xmin, F_n):
        self.F_n = F_n
        self.pn = pn
        self.dim = dim
        self.iter_max = iter_max
        self.w0 = 0.9
        self.w1 = 0.4
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.m = 7
        self.vmax = 100
        self.vmin = -100
        self.xmax = xmax
        self.xmin = xmin
        self.x = np.zeros((self.pn, self.dim))
        self.v = np.zeros((self.pn, self.dim))
        # 个体经历的最佳位置和全局最佳位置
        self.pbest = np.zeros((self.pn, self.dim))
        self.gbest = np.zeros((1, self.dim))
        self.pc = np.zeros(self.pn)
        self.flag = np.zeros(self.pn)
        self.f = np.zeros((self.pn, self.dim))
        # 每个个体的历史最佳适应值
        self.p_fit = np.zeros(self.pn)  # 每个粒子都有一个历史最佳适应值
        # 全局最佳适应值
        self.fit = 1e10

    # @staticmethod
    # def F_n(x):
    #     return np.sum(x**2)

    def init_population(self):
        self.x = self.xmin + (self.xmax - self.xmin) * np.random.rand(self.pn, self.dim)
        self.v = self.vmin + 2 * self.vmax * np.random.rand(self.pn, self.dim)
        self.pbest = self.x.copy()
        self.p_fit = np.array([self.F_n(self.x[i]) for i in range(self.x.shape[0])])
        self.fit = np.min(self.p_fit)
        self.gbest = self.x[np.argmin(self.p_fit)].copy()
        ind = np.array([i - 1 for i in range(self.pn)])
        self.pc = 0.05 + 0.45 * (np.exp((10 * ind / (self.pn - 1)) - 1) / (np.exp(10) - 1))

    def iterator(self):
        fitness = []
        for k in range(self.iter_max):
            w = self.w0 * ((self.w0 - self.w1) * k / self.iter_max)
            for i in range(self.pn):
                temp = self.F_n(self.x[i])
                if temp >= self.p_fit[i]:
                    self.flag[i] += 1
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pbest[i] = self.x[i].copy()
                    self.flag[i] = 0
                if self.p_fit[i] < self.fit:
                    self.gbest = self.x[i].copy()
                    self.fit = self.p_fit[i].copy()
            for i in range(self.pn):
                if self.flag[i] >= self.m:
                    for j in range(self.dim):
                        if random.random() < self.pc[i]:
                            f1 = int(np.floor(random.random() * self.pn))
                            f2 = int(np.floor(random.random() * self.pn))
                            if self.F_n(self.pbest[f1]) < self.F_n(self.pbest[f2]):
                                self.f[i][j] = f1
                            else:
                                self.f[i][j] = f2
                        else:
                            self.f[i][j] = i
                    self.flag[i] = 0
                else:
                    tt = random.randint(0, 32767)
                    for j in range(self.dim):
                        if j == tt % self.dim:
                            self.f[i][j] = tt % self.pn
                        else:
                            self.f[i][j] = i
            for i in range(self.pn):
                for j in range(self.dim):
                    self.v[i][j] = w * self.v[i][j] + self.c1 * random.random() * (self.pbest[int(self.f[i][j])][j] - self.x[i][j])
                    self.v[i][j] = np.clip(self.v[i][j], self.vmin, self.vmax)
                    self.x[i][j] = np.clip(self.x[i][j] + self.v[i][j], self.xmin, self.xmax)

            fitness.append(self.fit)
        return fitness



f_list1 = [resltestfunction.f1, resltestfunction.f2, resltestfunction.f3, resltestfunction.f4, resltestfunction.f5]  #单多峰

fitness_bound = {f_list1[0]: (-100, 100), f_list1[1]: (-100, 100), f_list1[2]: (-100, 100), f_list1[3]: (-15, 10), f_list1[4]: (-10, 10)}

dims=[100]
iter = 30
Maxiter = 500
all_f = [f_list1]
print("begin my iteration")
excel = pd.ExcelWriter('newtest-CLPSO-it500-d100.xlsx')
for ex_index, fn in enumerate(all_f):
    function_name = ["f" + str(i + 1) for i in range(len(fn))]
    df = pd.DataFrame(columns=['mean', 'std', 'min'], index=function_name)
    for index, f in enumerate(fn):
        result = []
        print("%d_%d processing" % (ex_index, index))
        for i in range(iter):
            [lower, upper] = fitness_bound[f]

            my_pso =CLPSO(pn=60, dim=dims[ex_index], iter_max=Maxiter, xmax=upper, xmin=lower, F_n=f)
            my_pso.init_population()
            fitness = my_pso.iterator()
            # print(np.min(fitness), np.mean(fitness), np.std(fitness))
            # print("---------------------------------------")
            result.append(np.min(fitness))
            if i == iter - 1:
                df.loc[function_name[index], 'mean'] = np.mean(result)
                df.loc[function_name[index], 'std'] = np.std(result)
                df.loc[function_name[index], 'min'] = np.min(result)
        # result[index, :] = [np.min(fitness), np.mean(fitness), np.std(fitness)]
        df.to_excel(excel, sheet_name=str(ex_index + 1))
        excel.save()
