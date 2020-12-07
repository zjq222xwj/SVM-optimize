import pandas as pd
import numpy.random as rd


def loss(x):
	data = pd.read_excel('C:/Users/10429/Desktop/案例数据.xls')
	A = data['A']
	D = data['D']
	Z = data['锌锭总耗量']
	J = 0
	n = len(data)
	for i in range(n):
		J += (A[i] - Z[i] * x) * (A[i] - Z[i] * x) + (D[i] - Z[i] * (1 - x)) * (D[i] - Z[i] * (1 - x));
		if (x > 1 or x < 0):
			J += 1e10
		n = n+1
	return 0.5*J


def dx(x):
	data = pd.read_excel('C:/Users/10429/Desktop/案例数据.xls')
	A = data['A']
	D = data['D']
	Z = data['锌锭总耗量']
	a = 0
	n = len(data)
	for i in range(n):
		a += -Z[i] * (A[i] - Z[i] * x) + Z[i] * (D[i] - Z[i] * (1 - x));
		n += 1
	return a


def Gradient_descent(initial_x, max_iters, eta, err = 0.0001):
	x = initial_x
	i_iter = 0
	while i_iter  < max_iters:
		delta_x = dx(x)
		print('dx',delta_x)
		last_x = x;
		x = x - eta * delta_x;
		if (abs(loss(x) - loss(last_x)) < err):
			break
		# if (x > 1 or y > 1 or x < 0 or y < 0):
		# 	break
		print("第", i_iter, "次迭代：x 值为 ", x)
		i_iter += 1;
	return x

if __name__ == '__main__':
	initial_x = rd.random()
	max_iters = 100
	eta = 1e-8
	err = 1e-6
	best_x = Gradient_descent(initial_x, max_iters, eta, err)
	print(best_x)