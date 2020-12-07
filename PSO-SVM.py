# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm
import math
import random as rd





def svr(x):
    # x输入粒子位置
    # y 粒子适应度值
    rbf_svm = svm.SVR(kernel='rbf', C=x[0], gamma=x[1])  # svm的使用函数
    # SVM模型预测及其精度
    cv_scores = model_selection.cross_val_score(rbf_svm, X_train_scaled, Y_train, cv=5, scoring='neg_mean_squared_error')
    # 以错误率最小化为目标
    scores = (-cv_scores).mean()
    # fitness = (1 - scores) * 100
    fitness = scores
    return fitness


def initpopvfit(size):
    pop = np.zeros((size,2))
    v = np.zeros((size,2))
    fitness = np.zeros(size)

    for i in range(size):
        pop[i] = [rangepop[0]+(rangepop[1]-rangepop[0])*np.random.rand(),rangepop[0]+(rangepop[1]-rangepop[0])*np.random.rand()]
        v[i] = [rd.uniform(-1,1),rd.uniform(-1,1)]
        fitness[i] = svr(pop[i])
    return pop,v,fitness

def getinitbest(fitness,pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop,gbestfitness = pop[fitness.argmin()].copy(),fitness.min()
    #个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
    pbestpop,pbestfitness = pop.copy(),fitness.copy()

    return gbestpop,gbestfitness,pbestpop,pbestfitness

w =  1
# C1= 0.49445
# C2= 1.49445
C1= 2
C2= 2
Max_iteration = 50
size = 20
rangepop= (0.01, 100)
rangepopc = (0.1, 100)#粒子位置限制
rangepopg = (0.01,1)
rangespeed = (-0.5,0.5)  #速度限制

data = pd.read_excel('C:/Users/10429/Desktop/data/708daylyCulled.xls')
data = data.dropna(how='any',axis=0)
xlab = ['cw', 'targetZnc', 'cw-plat-01', 'cw-plat-null', 'targetZnc-plat-01', 'targetZnc-plat-null']
ylab = ['totalZnc']
X = data[xlab]
Y = data[ylab]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
Y_train = Y_train.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1, 1)
Y_train = Y_train.ravel()
Y_test = Y_test.ravel()
X_scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

pop,v,fitness = initpopvfit(size)
print(pop)
print(v)
print(fitness)
gbestpop,gbestfitness,pbestpop,pbestfitness = getinitbest(fitness,pop)
print(gbestfitness)

mse = []
for i in range(Max_iteration):
        t=0.5
        #速度更新
        for j in range(size):
            v[j] += C1*np.random.rand()*(pbestpop[j]-pop[j])+C2*np.random.rand()*(gbestpop-pop[j])
        v[v<rangespeed[0]] = rangespeed[0]
        v[v>rangespeed[1]] = rangespeed[1]

        #粒子位置更新
        for j in range(size):
            #pop[j] += 0.5*v[j]
            # pop[j] = t*(0.5*v[j])+(1-t)*pop[j]
            pop[j] += v[j]
        pop[pop<rangepop[0]] = rangepop[0]
        pop[pop>rangepop[1]] = rangepop[1]

        #适应度更新
        for j in range(size):
            fitness[j] = svr(pop[j])

        for j in range(size):
            if fitness[j] < pbestfitness[j]:
                pbestfitness[j] = fitness[j]
                pbestpop[j] = pop[j].copy()

        if pbestfitness.min() < gbestfitness :
            gbestfitness = pbestfitness.min()
            gbestpop = pop[pbestfitness.argmin()].copy()

        print('----------------迭代次数--------------------' + str(i+1))
        print('C and gamma:' + str(gbestpop))
        print('mse:' + str(gbestfitness))
        mse.append(gbestfitness)


plt.plot(mse)
plt.xlabel('Number of iteration', size=10)
plt.ylabel('mse', size=10)
plt.title('PSO_RBF_SVR parameter optimization')
plt.show()

bestC = gbestpop[0]
bestgamma = gbestpop[1]
print("The best C is " + str(bestC))
print("The best gamma is " + str(bestgamma))
model_rbf_svm = svm.SVR(kernel='rbf', C=bestC, gamma=bestgamma).fit(X_train_scaled, Y_train)
Y_pred = model_rbf_svm.predict(X_test_scaled)
print(mean_squared_error(Y_test,Y_pred))