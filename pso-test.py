# coding: utf-8
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):  # 初始化类  设置粒子数量  位置信息维度  最大迭代次数
        self.w = 0.8
        self.c1 = 1.5
        self.c2 = 1.5
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置（还要确定取值范围）
        self.Xmax = 100
        self.Xmin = -100
        self.V =  np.zeros((self.pN, self.dim))  # 所有粒子的速度（还要确定取值范围）
        self.Vmax = 10
        self.Vmin = -10
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置
        self.gbest = np.zeros((1, self.dim))  # 全局最佳位置
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = float(1e10)  # 全局最佳适应值

    # ---------------------目标函数Sphere函数-----------------------------
    def function(self, x):
        f = 0
        for i in range(self.dim):
            f += x[i] ** 2
        return f

    # rastrigin测试函数[-5.12,5.12]
    # def function(self, x):
    #     A = 10.0
    #     f = 0
    #     for i in range(self.dim):
    #         f += x[i] ** 2 - A * math.cos(2 * math.pi * x[i]) + A
    #     return f

    # Ackley测试函数[-32,32]
    # def function(self, x):
    #     a = 20
    #     b = 0.2
    #     c = 2.0 * math.pi
    #     firstSum = 0.0
    #     secondSum = 0.0
    #     for i in range(self.dim):
    #         firstSum += x[i] ** 2.0
    #         secondSum += math.cos(c * x[i])
    #     n = float(self.dim)
    #     f = -a * math.exp(-b * math.sqrt(firstSum / n)) - math.exp(secondSum / n) + a + math.e
    #     return f
    # ---------------------初始化种群----------------------------------
    def init_Population(self):

        for i in range(self.pN):  # 遍历所有粒子

            for j in range(self.dim):  # 每一个粒子的纬度
                self.X[i][j] = random.uniform(-100, 100)  # 给每一个粒子的位置赋一个初始随机值（在一定范围内）
                self.V[i][j] = random.uniform(-0.1, 0.1)  # 给每一个粒子的速度给一个初始随机值（在一定范围内）

            self.pbest[i] = self.X[i]  # 把当前粒子位置作为这个粒子的最优位置

            tmp = self.function(self.X[i])  # 计算这个粒子的适应度值

            self.p_fit[i] = tmp  # 当前粒子的适应度值作为个体最优值

            if (tmp < self.fit):  # 与当前全局最优值做比较并选取更佳的全局最优值

                self.fit = tmp
                self.gbest = self.X[i]


     # ---------------------更新粒子位置----------------------------------

    def iterator(self):

        fitness = []

        for t in range(self.max_iter):

            for i in range(self.pN):

                # 更新速度
                self.V[i] = self.w * self.V[i] + self.c1 * random.uniform(0,1) * (self.pbest[i] - self.X[i]) + (self.c2 * random.uniform(0,1) * (self.gbest - self.X[i]))

                for j in range(self.dim):
                    if self.V[i][j] > self.Vmax:
                        self.V[i][j] = self.Vmax
                    elif self.V[i][j] < self.Vmin:
                        self.V[i][j] = self.Vmin

                # 更新位置
                self.X[i] = self.X[i] + self.V[i]

                for j in range(self.dim):
                    if self.X[i][j] > self.Xmax:
                        self.X[i][j] = self.Xmax
                    elif self.X[i][j]  < self.Xmin:
                        self.X[i][j]  = self.Xmin

            for i in range(self.pN):  # 更新gbest\pbest

                temp = self.function(self.X[i])

                if (temp < self.p_fit[i]):  # 更新个体最优
                    self.pbest[i] = self.X[i]
                    self.p_fit[i] = temp

                if (temp < self.fit):  # 更新全局最优
                    self.gbest = self.X[i]
                    self.fit = temp

            fitness.append(self.fit)
            print(t, self.fit)  # 输出最优值

        return fitness


# ----------------------程序执行-----------------------
if __name__ == '__main__':
    fit_g = []
    for i in range(30):
        my_pso = PSO(pN=30, dim=30, max_iter=500)
        my_pso.init_Population()
        my_pso.iterator()
        fit_g.append(my_pso.fit)
    fig_mean = np.mean(fit_g)
    fig_std = np.std(fit_g, ddof=1)
    print("最优值为：%f" % min(fit_g))
    print("平均值为：%f" % fig_mean)
    print("最差值为：%f" % max(fit_g))
    print("标准差为:%f" % fig_std)
    print(fit_g)

