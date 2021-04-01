import numpy as np
import resltestfunction
import matplotlib.pyplot as plt
import pandas as pd


class PSO:
    def __init__(self, pn, dim, max_iter, x_lower, x_upper, F_n):
        self.F_n = F_n
        self.pn = pn
        self.dim = dim
        self.max_iter = max_iter
        self.max_func = 11
        self.w = 0.7298
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.x_lower = x_lower  # 解空间范围的下界
        self.x_upper = x_upper  # 解空间的上界
        self.v_lower = x_lower * 0.2
        self.v_upper = x_upper * 0.2
        self.x = np.zeros((self.pn, self.dim))
        self.v = np.zeros((self.pn, self.dim))
        # 个体经历的最佳位置和全局最佳位置
        self.pbest = np.zeros((self.pn, self.dim))
        self.gbest = np.zeros(self.dim)
        # 每个个体的历史最佳适应值
        self.p_fit = np.zeros(self.pn)  # 每个粒子都有一个历史最佳适应值
        # 全局最佳适应值
        self.fit = 0
        self.data = []

    def init_Population(self):
        self.x = np.random.rand(self.pn, self.dim) * (self.x_upper - self.x_lower) + self.x_lower
        self.v = self.v_lower + 2 * self.v_upper * np.random.rand(self.pn, self.dim)
        self.p_fit = np.array([self.F_n(self.x[i]) for i in range(self.x.shape[0])])
        self.pbest = self.x.copy()
        self.fit = np.min(self.p_fit)
        self.gbest = self.x[np.argmin(self.p_fit)]

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):  # 迭代循环
            for i in range(self.pn):
                temp = self.F_n(self.x[i])
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pbest[i] = self.x[i].copy()
                if self.p_fit[i] < self.fit:
                    self.gbest = self.x[i].copy()
                    self.fit = self.p_fit[i].copy()
            self.v = self.w * self.v + self.c1 * np.random.rand(self.pn, self.dim) * (self.pbest - self.x) +\
                        self.c2 * np.random.rand(self.pn, self.dim) * (self.gbest - self.x)
            self.v = np.clip(self.v, self.v_lower, self.v_upper)
            self.x = np.clip(self.x + self.v, self.x_lower, self.x_upper)
            fitness.append(self.fit)
        return fitness


f_list1 = [resltestfunction.f1, resltestfunction.f2, resltestfunction.f3, resltestfunction.f4, resltestfunction.f5]  #单多峰

fitness_bound = {f_list1[0]: (-100, 100), f_list1[1]: (-100, 100), f_list1[2]: (-100, 100), f_list1[3]: (-15, 10), f_list1[4]: (-10, 10)}

dims=[100]
iter = 30
Maxiter = 500
all_f = [f_list1]
print("begin my iteration")
excel = pd.ExcelWriter('newtest-PSO-it500-d100.xlsx')
for ex_index, fn in enumerate(all_f):
    function_name = ["f" + str(i + 1) for i in range(len(fn))]
    df = pd.DataFrame(columns=['mean', 'std', 'min'], index=function_name)
    for index, f in enumerate(fn):
        result = []
        print("%d_%d processing" % (ex_index, index))
        for i in range(iter):
            [lower, upper] = fitness_bound[f]
            my_pso = PSO(pn=60, dim=dims[ex_index], x_lower=-50, x_upper=50, max_iter=Maxiter,  F_n=f)
            my_pso.init_Population()
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

