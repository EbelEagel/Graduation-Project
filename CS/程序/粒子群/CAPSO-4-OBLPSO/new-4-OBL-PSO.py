import numpy as np
import random
import matplotlib.pyplot as plt
import resltestfunction
import math
import pandas as pd


class OBLCPSO:
    def __init__(self, pn, dim, x_lower, x_upper, max_gen, F_n):
        self.pn = pn
        self.dim = dim
        self.F_n = F_n
        self.max_gen = max_gen
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.v_lower = x_lower * 0.2
        self.v_upper = x_upper * 0.2
        self.x = np.zeros((self.pn, self.dim))
        self.v = np.zeros((self.pn, self.dim))
        self.winner = np.zeros((self.pn//3, self.dim))
        self.loser = np.zeros((self.pn//3, self.dim))
        self.v_loser = np.zeros((self.pn//3, self.dim))
        self.mid = np.zeros((self.pn//3, self.dim))
        self.mean = np.zeros(self.dim)
        self.pbest = np.zeros((self.pn, self.dim))
        self.gbest = np.zeros(self.dim)
        self.c_val = np.zeros((3, self.dim))
        self.p_fit = np.zeros(self.pn)
        self.pn_id = np.zeros(self.pn)
        self.mid_id = np.zeros(self.pn//3, dtype=int)
        self.fit = 1e10

    def init_population(self):
        self.x = self.x_lower + (self.x_upper - self.x_lower) * np.random.rand(self.pn, self.dim)
        self.v = self.v_lower + 2 * self.v_upper * np.random.rand(self.pn, self.dim)
        self.mean = np.mean(self.x, axis=0)
        self.p_fit = np.array([self.F_n(self.x[i]) for i in range(self.x.shape[0])])
        self.pbest = self.x.copy()
        self.fit = np.min(self.p_fit)
        self.pn_id = [ii for ii in range(self.pn)]
        random.shuffle(self.pn_id)

    def iteration(self):
        fitness = []
        t = 0
        n = self.pn//3
        while t < self.max_gen:
            fitness.append(self.fit)
            w = (0.7 - 0.2) * (self.max_gen - t) / self.max_gen + 0.2
            for k in range(n):
                r = np.zeros((3, self.dim))
                r[0] = self.x[self.pn_id[k]].copy()
                r[1] = self.x[self.pn_id[k + n]].copy()
                r[2] = self.x[self.pn_id[k + 2*n]].copy()
                temp = np.array([self.F_n(r[0]), self.F_n(r[1]), self.F_n(r[2])])
                self.winner[k] = r[np.argmin(temp)].copy()
                self.loser[k] = r[np.argmax(temp)].copy()
                self.v_loser[k] = self.v[self.pn_id[k + np.argmax(temp) * n]].copy()
                for i in range(3):
                    if i != np.argmin(temp) and (i != np.argmax(temp)):
                        self.mid_id[k] = i
                        self.mid[k] = r[self.mid_id[k]].copy()

                self.v_loser[k] = np.random.rand(self.dim) * self.v_loser[k] + np.random.rand(self.dim) * (self.winner[k]
                 - self.loser[k]) + w * np.random.rand(self.dim) * (self.mean - self.loser[k])
                self.v[self.pn_id[k + np.argmax(temp) * n]] = self.v_loser[k].copy()
                self.loser[k] = self.loser[k] + self.v_loser[k]
                self.x[self.pn_id[k + np.argmax(temp) * n]] = self.loser[k].copy()
                self.mid[k] = self.x_upper + self.x_lower - self.mid[k] + np.random.rand(self.dim) * self.mid[k]
                self.x[k + self.mid_id[k] * n] = self.mid[k].copy()

                LN_min = min(self.F_n(self.winner[k]), self.F_n(self.loser[k]), self.F_n(self.mid[k]))
                if LN_min < self.fit:
                    self.fit = LN_min
            self.mean = np.mean(self.x, axis=0)
            t += 1

        return fitness


f_list1 = [resltestfunction.f1, resltestfunction.f2, resltestfunction.f3, resltestfunction.f4, resltestfunction.f5]  #单多峰

fitness_bound = {f_list1[0]: (-100, 100), f_list1[1]: (-100, 100), f_list1[2]: (-100, 100), f_list1[3]: (-15, 10), f_list1[4]: (-10, 10)}

dims=[100]
iter = 30
Maxiter = 500
all_f = [f_list1]
print("begin my iteration")
excel = pd.ExcelWriter('newtest-OBLPSO-it500-d100.xlsx')
for ex_index, fn in enumerate(all_f):
    function_name = ["f" + str(i + 1) for i in range(len(fn))]
    df = pd.DataFrame(columns=['mean', 'std', 'min'], index=function_name)
    for index, f in enumerate(fn):
        result = []
        print("%d_%d processing" % (ex_index, index))
        for i in range(iter):
            [lower, upper] = fitness_bound[f]
            my_pso = OBLCPSO(pn=60, dim=dims[ex_index], x_lower=lower, x_upper=upper, max_gen=Maxiter, F_n=f)
            my_pso.init_population()
            fitness = my_pso.iteration()
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
