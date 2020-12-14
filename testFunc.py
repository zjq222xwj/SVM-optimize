# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from sklearn import svm
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy.random as rd
import matplotlib.pyplot as plt


#  适应度函数
# Sphere测试函数[-100,100]
def sphere(position):
    total = 0
    for i in range(len(position)):
        total += position[i] ** 2
    return total


# Ackley测试函数[-32,32]
def ackley(position):
    a = 20
    b = 0.2
    c = 2.0 * math.pi
    firstSum = 0.0
    secondSum = 0.0
    for i in range(len(position)):
        firstSum += position[i] ** 2.0
        secondSum += math.cos(c * position[i])
    n = float(len(position))
    f = -a * math.exp(-b * math.sqrt(firstSum / n)) - math.exp(secondSum / n) + a + math.e
    return f


#  rastrigin测试函数[-5.12,5.12]
def Rastrigin(position):
    A = 10.0
    total = 0
    for i in range(len(position)):
        total += position[i] ** 2 - A * math.cos(2 * math.pi * position[i]) + A
    return total


#  rosenbrock测试函数[-30,30]
def Rosenbrock(position):
    a = 100.0
    total = 0
    for i in range(len(position)-1):
        total += a*(position[i+1] - position[i] * position[i])**2 + (position[i]-1)*(position[i]-1)
    return total


#   生成tent混沌序列
def tentmap12(alpha, x0, max_g):
    """
    tent map 函数
    :param alpha:0到1之间的参数
    :param x0: 初值
    :param max_g: 迭代次数
    :return: 函数值
    """
    x = x0
    x_list = []
    for i in max_g:
        if x < alpha:
            x = x / alpha
        else:
            x = (1 - x) / (1 - alpha)
        x_list.append(x)
    return x_list

#  超过搜索空间，边界处理
def relocation(loc, lb, ub):
    if loc > ub:
        loc = ub
    if loc < lb:
        loc = lb
    return loc

# 3. GWO优化算法
def IGWO(fitfunc,SearchAgents_no, Max_iteration, dim, lb, ub):
    # 初始化头狼的位置
    Alpha_pos = np.zeros((SearchAgents_no,dim))
    Beta_pos = np.zeros((SearchAgents_no,dim))
    Delta_pos = np.zeros((SearchAgents_no,dim))
    # 初始化Alpha狼的目标函数值
    Alpha_score = float("inf")
    Beta_score = float("inf")
    Delta_score = float("inf")

    # TO DO tent映射生成初始种群
    max_g = np.linspace(1, SearchAgents_no, num=SearchAgents_no)
    # 生成tent序列 30维度
    x_list = np.zeros((dim, SearchAgents_no))
    tent = np.zeros((SearchAgents_no, dim))
    for i in range(0, dim):
        x_list[i] = tentmap12(0.5, rd.random(1), max_g)
    for i in range(0, SearchAgents_no):
        # 若搜索位置超过了搜索空间，需要重新回到搜索空间
        for j in range(0, dim):
            tent[i,j] = x_list[j,i]
    Positions = np.dot(tent, (ub - lb)) + lb
    # Positions = np.dot(rd.rand(SearchAgents_no, dim), (ub - lb)) + lb  # 初始化首次搜索位置
    print('-------')

    iterations = []
    f = []

    # 主循环
    index_iteration = 0
    while index_iteration < Max_iteration:

        # 遍历每个狼
        for i in range(0, SearchAgents_no):
            # 若搜索位置超过了搜索空间，需要重新回到搜索空间
            for j in range(0, dim):
                Positions[i, j] = relocation(Positions[i, j], lb, ub)
            scores = fitfunc(Positions[i])
            # fitness = (1 - scores) * 100
            fitness = scores
            if fitness < Alpha_score:  # 如果目标函数值小于Alpha狼的目标函数值
                Alpha_score = fitness  # 则将Alpha狼的目标函数值更新为最优目标函数值
                Alpha_pos = Positions[i]  # 同时将Alpha狼的位置更新为最优位置
            if fitness > Alpha_score and fitness < Beta_score:  # 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
                Beta_score = fitness  # 则将Beta狼的目标函数值更新为最优目标函数值
                Beta_pos = Positions[i]
            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:  # 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
                Delta_score = fitness  # 则将Delta狼的目标函数值更新为最优目标函数值
                Delta_pos = Positions[i]

        #  TO DO 收敛因子改进
        m=rd.random(1)
        a = 2 *((1-index_iteration / Max_iteration)**m)
        # a = 2 - index_iteration * (2 / Max_iteration)

        # 遍历每个狼
        for i in range(0, SearchAgents_no):
            # 遍历每个维度
            for j in range(0, dim):
                # 包围猎物，位置更新
                r1 = rd.random(1)  # 生成0~1之间的随机数
                r2 = rd.random(1)
                A1 = 2 * a * r1 - a  # 计算系数A
                C1 = 2 * r2  # 计算系数C

                # Alpha狼位置更新
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = rd.random(1)
                r2 = rd.random(1)

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                # Beta狼位置更新
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = rd.random(1)
                r2 = rd.random(1)

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                # Delta狼位置更新
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                # 位置更新
                Positions[i, j] = (X1 + X2 + X3) / 3
                # Positions[i, j] = (5 * X1 + 3 * X2 + 2 * X3) / 10

        # TO DO差分进化
        # 交叉
        # Wmax = 1.5
        # Wmin = 0.25
        # W = (Wmax - Wmin) * (Max_iteration - index_iteration) / Max_iteration + Wmin
        W = 0.5
        V = Alpha_pos + W * (Beta_pos - Delta_pos)
        # 变异
        CR = 0.7  # 交叉概率常数
        U = [[0 for i in range(dim)] for i in range(SearchAgents_no)]
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                # 判断变异是否满足边界
                V[j] = relocation(V[j], lb, ub)
                # 更新狼的位置边界处理
                Positions[i, j] = relocation(Positions[i, j], lb, ub)
                rand_j = rd.randint(0, dim - 1)
                rand_float = rd.random()
                if rand_float <= CR or rand_j == j:
                    U[i][j] = V[j]
                else:
                    U[i][j] = Positions[i, j]
        # 选择
        for i in range(0, SearchAgents_no):
            # 重计算适应度函数
            x_score = fitfunc(Positions[i])
            u_score = fitfunc(U[i])
            if u_score <= x_score:
                Positions[i] = U[i]

        index_iteration = index_iteration + 1
        iterations.append(index_iteration)
        # accuracy.append((100 - Alpha_score) / 100)
        f.append(Alpha_score)
        print('----------------迭代次数--------------------' + str(index_iteration))
        print('f:' + str(Alpha_score))

    return iterations, f


def GWO(fitfunc, SearchAgents_no, Max_iteration, dim, lb, ub):
    # 初始化头狼的位置
    Alpha_pos = np.zeros((SearchAgents_no,dim))
    Beta_pos = np.zeros((SearchAgents_no,dim))
    Delta_pos = np.zeros((SearchAgents_no,dim))

    # 初始化Alpha狼的目标函数值
    Alpha_score = float("inf")
    Beta_score = float("inf")
    Delta_score = float("inf")

    # 初始化首次搜索位置
    # TO DO tent映射生成初始种群
    max_g = np.linspace(1, SearchAgents_no, num=SearchAgents_no)
    # 生成tent序列 30维度
    x_list = np.zeros((dim, SearchAgents_no))
    tent = np.zeros((SearchAgents_no, dim))
    for i in range(0, dim):
        x_list[i] = tentmap12(0.6, rd.random(1), max_g)
    for i in range(0, SearchAgents_no):
        # 若搜索位置超过了搜索空间，需要重新回到搜索空间
        for j in range(0, dim):
            tent[i, j] = x_list[j, i]
    Positions = np.dot(tent, (ub - lb)) + lb
    # Positions = np.dot(rd.rand(SearchAgents_no, dim), (ub - lb)) + lb
    print(Positions)
    print('-------')

    iterations = []
    f = []

    # 主循环
    index_iteration = 0
    while index_iteration < Max_iteration:

        # 遍历每个狼
        for i in range(0, SearchAgents_no):
            # 若搜索位置超过了搜索空间，需要重新回到搜索空间
            for j in range(0, dim):
                Positions[i, j] = relocation(Positions[i, j], lb, ub)
            scores = fitfunc(Positions[i])
            # fitness = (1 - scores) * 100
            fitness = scores
            if fitness < Alpha_score:  # 如果目标函数值小于Alpha狼的目标函数值
                Alpha_score = fitness  # 则将Alpha狼的目标函数值更新为最优目标函数值
                Alpha_pos = Positions[i]  # 同时将Alpha狼的位置更新为最优位置
            if fitness > Alpha_score and fitness < Beta_score:  # 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
                Beta_score = fitness  # 则将Beta狼的目标函数值更新为最优目标函数值
                Beta_pos = Positions[i]
            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:  # 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
                Delta_score = fitness  # 则将Delta狼的目标函数值更新为最优目标函数值
                Delta_pos = Positions[i]

        a = 2 - index_iteration * (2 / Max_iteration)

        # 遍历每个狼
        for i in range(0, SearchAgents_no):
            # 遍历每个维度
            for j in range(0, dim):
                # 包围猎物，位置更新
                r1 = rd.random(1)  # 生成0~1之间的随机数
                r2 = rd.random(1)
                A1 = 2 * a * r1 - a  # 计算系数A
                C1 = 2 * r2  # 计算系数C

                # Alpha狼位置更新
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = rd.random(1)
                r2 = rd.random(1)

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                # Beta狼位置更新
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = rd.random(1)
                r2 = rd.random(1)

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                # Delta狼位置更新
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                # 位置更新
                Positions[i, j] = (X1 + X2 + X3) / 3

        index_iteration = index_iteration + 1
        iterations.append(index_iteration)
        f.append(Alpha_score)
        print('----------------迭代次数--------------------' + str(index_iteration))
        print('f:' + str(Alpha_score))
    return iterations, f


def plot(iterations, mse):
    plt.plot(iterations, mse,c='green')
    plt.xlabel('iteration', size=14)
    plt.ylim(0,10000)
    plt.ylabel('best score', size=10)
    plt.title('GWO')
    plt.show()


if __name__ == '__main__':
    print('----------------.参数设置------------')
    SearchAgents_no = 30  # 狼群数量
    Max_iteration = 500  # 最大迭代次数
    dim = 30  # 维度
    lb = -100  # 参数取值下界
    ub = 100  # 参数取值上界

    print('----------------3.GWO-----------------')
    # iterations, f = GWO(sphere,SearchAgents_no, Max_iteration, dim, lb, ub)
    # Iiterations, If = IGWO(Rastrigin, SearchAgents_no, Max_iteration, dim, lb, ub)
    print('----------------4.结果显示-----------------')
    # plt.plot(iterations, f)
    # plt.plot(Iiterations, If, c='green')
    #     # plt.xlabel('iteration', size=14)
    #     # plt.ylim(0, 20)
    #     # plt.ylabel('best score', size=10)
    #     # plt.title('GWO')
    #     # plt.show()
    fit_g = []
    for i in range(3):
        iterations, f = GWO(sphere,SearchAgents_no, Max_iteration, dim, lb, ub)
        fit_g.append(f[-1])
    fig_mean = np.mean(fit_g)
    fig_std = np.std(fit_g, ddof=1)
    print("最优值为：%f" % min(fit_g))
    print("平均值为：%f" % fig_mean)
    print("最差值为：%f" % max(fit_g))
    print("标准差为:%f" % fig_std)
    print(min(fit_g))
    print(fig_mean)
    print(max(fit_g))
    print(fig_std)
    print(fit_g)




