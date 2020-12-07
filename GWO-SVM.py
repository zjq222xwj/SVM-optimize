# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy.random as rd
import matplotlib.pyplot as plt


## 1.加载数据
def load_data(data_file,xlab,ylab):
    data = pd.read_excel(data_file)
    data = data.dropna(how='any', axis=0)
    X = data[xlab]
    Y = data[ylab]
    return np.array(X),np.array(Y)


## 2. GWO优化算法
def gwo(X_train_scaled,  Y_train, X_test_scaled, Y_test,SearchAgents_no, Max_iteration, dim, lb, ub):
    Alpha_pos = [0, 0]  # 初始化Alpha狼的位置
    Beta_pos = [0, 0]
    Delta_pos = [0, 0]

    Alpha_score = float("inf")  # 初始化Alpha狼的目标函数值
    Beta_score = float("inf")
    Delta_score = float("inf")

    Positions = np.dot(rd.rand(SearchAgents_no, dim), (ub - lb)) + lb  # 初始化首次搜索位置
    print('初始位置')
    print(Positions)
    print(Positions.shape[0])
    print(Positions.shape[1])
    Convergence_curve = np.zeros((1, Max_iteration))  # 初始化融合曲线

    iterations = []
    mse = []

    # 主循环
    index_iteration = 0
    while index_iteration < Max_iteration:

        # 遍历每个狼
        for i in range(0, (Positions.shape[0])):
            # 若搜索位置超过了搜索空间，需要重新回到搜索空间
            for j in range(0, (Positions.shape[1])):
                Flag4ub = Positions[i, j] > ub
                Flag4lb = Positions[i, j] < lb
                # 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界
                if Flag4ub:
                    Positions[i, j] = ub
                if Flag4lb:
                    Positions[i, j] = lb
            # SVM模型训练
            rbf_svm = svm.SVR(kernel='rbf', C=Positions[i][0], gamma=Positions[i][1])  # svm的使用函数
            # SVM模型预测及其精度
            cv_scores = model_selection.cross_val_score(rbf_svm, X_train_scaled, Y_train, cv=5, scoring='neg_mean_squared_error')
            # 以错误率最小化为目标
            scores = (-cv_scores).mean()
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
        for i in range(0, (Positions.shape[0])):
            # 遍历每个维度
            for j in range(0, (Positions.shape[1])):
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
        # accuracy.append((100 - Alpha_score) / 100)
        mse.append(Alpha_score)
        print('----------------迭代次数--------------------' + str(index_iteration))
        print(Positions)
        print('C and gamma:' + str(Alpha_pos))
        # print('accuracy:' + str((100 - Alpha_score) / 100))
        print('mse:' + str(Alpha_score))

    bestC = Alpha_pos[0]
    bestgamma = Alpha_pos[1]

    return bestC, bestgamma, iterations, mse


def plot(iterations, mse):
    plt.plot(iterations, mse)
    plt.xlabel('Number of iteration', size=20)
    plt.ylabel('mse', size=20)
    plt.title('GWO_RBF_SVR parameter optimization')
    plt.show()


if __name__ == '__main__':
    print('----------------1.加载数据-------------------')
    data_file = 'C:/Users/10429/Desktop/data/708daylyCulled.xls'
    xlab = ['cw', 'targetZnc', 'cw-plat-01', 'cw-plat-null', 'targetZnc-plat-01', 'targetZnc-plat-null']
    ylab = ['totalZnc']
    X, Y = load_data(data_file,xlab,ylab)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    Y_train = Y_train.reshape(-1,1)
    Y_test = Y_test.reshape(-1, 1)
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()
    X_scaler = preprocessing.MinMaxScaler().fit(X_train)
    # Y_scaler = preprocessing.MinMaxScaler().fit(Y_train)

    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    # Y_train_scaled = Y_scaler.transform(Y_train)
    # Y_test_scaled = Y_scaler.transform(Y_test)
    # Y_train_scaled = Y_train_scaled.ravel()
    # Y_test_scaled = Y_test_scaled.ravel()
    print('----------------2.参数设置------------')
    SearchAgents_no = 20  # 狼群数量
    Max_iteration = 50  # 最大迭代次数
    dim = 2  # 需要优化两个参数c和g
    lb = 0.01  # 参数取值下界
    ub = 100  # 参数取值上界

    print('----------------3.GWO-----------------')
    bestC, bestgamma, iterations, mse = gwo(X_train_scaled, Y_train,X_test_scaled, Y_test,SearchAgents_no,Max_iteration, dim, lb, ub)
    print('----------------4.结果显示-----------------')
    print("The best C is " + str(bestC))
    print("The best gamma is " + str(bestgamma))
    plot(iterations, mse)

    model_rbf_svm = svm.SVR(kernel='rbf', C=bestC, gamma=bestgamma).fit(X_train_scaled, Y_train)
    Y_pred = model_rbf_svm.predict(X_test_scaled)
    print(mean_squared_error(Y_test,Y_pred))


