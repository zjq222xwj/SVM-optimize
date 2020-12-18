import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
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


def draw12(alpha, x0, iters):
    """
    画出tent map 的图像
    :param alpha:0到1之间的参数
    :param x0:初值
    :param iters:迭代次数
    :return:图像
    """
    max_g = np.linspace(1, iters, num=iters)
    x_list = tentmap12(alpha, x0, max_g)
    print(x_list[50])
    plt.plot(max_g, x_list)
    plt.title('tent map')
    plt.xlabel('iters')
    plt.ylabel('x')
    plt.xlim(0, 500)
    plt.ylim(0, 1)
    plt.show()


draw12(0.6, rd.random(1), 500)