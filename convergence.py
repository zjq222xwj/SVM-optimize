import numpy.random as rd
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False  #显示负号
Max_iteration = 500
index_iteration = 0
iterations = []
arr = []
airr = []
while index_iteration < Max_iteration:
    m=rd.random(1)
#     ai = 2 *((1-index_iteration / Max_iteration)**m)
    ai = 2 - 2 * (index_iteration / Max_iteration)**2
#     ai = 2 * math.cos(index_iteration / Max_iteration)
    a = 2 - 2 * (index_iteration / Max_iteration)
    index_iteration = index_iteration + 1
    iterations.append(index_iteration)
    arr.append(a)
    airr.append(ai)

plt.figure(figsize=(6.4,4.8), dpi=100)
plt.grid(linestyle = "--")      #设置背景网格线为虚线
ax = plt.gca()
# ax.spines['top'].set_visible(False)  #去掉上边框
# ax.spines['right'].set_visible(False) #去掉右边框
plt.plot(iterations, arr,label='原始a',linewidth=1.5)
plt.plot(iterations, airr,label='改进a',linestyle='--',linewidth=1.5)
plt.xticks(fontsize=12,fontweight='bold') #默认字体大小为10
plt.yticks(fontsize=12,fontweight='bold')
# plt.title("example",fontsize=12,fontweight='bold')   #默认字体大小为12
plt.xlabel("Data sets",fontsize=13,fontweight='bold')
plt.ylabel("Accuracy",fontsize=13,fontweight='bold')
plt.legend(prop={'size': 12})
plt.xlabel('迭代次数', fontsize=13,fontweight='bold')
plt.ylabel('收敛因子', fontsize=13,fontweight='bold')
plt.ylim(0, 2)
plt.xlim(0, 500)
# plt.legend(loc=0, numpoints=1)
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=12,fontweight='bold') #设置图例字体的大小和粗细
# plt.grid()
plt.show()