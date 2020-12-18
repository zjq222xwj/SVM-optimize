import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# sphere
# X = np.arange(-100, 100, 0.1)
# Y = np.arange(-100, 100, 0.1)
# X, Y = np.meshgrid(X, Y)
# Z = X**2 + Y**2
# fig = plt.figure(figsize=(8, 6))
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('F1(X1,X2)')
# ax.set_zlim(0,2.0e4)
# ax.set_title('Sphere Function')
# plt.show()

#ackley
# X = np.arange(-32, 32, 0.1)
# Y = np.arange(-32, 32, 0.1)
# X, Y = np.meshgrid(X, Y)
# Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2))) - \
#         np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20
# fig = plt.figure(figsize=(8,6))
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('F2(X1,X2)')
# ax.set_zlim(0,20)
# # ax.set_title('Ackley Function')
# plt.show()


#rastrigin
# X = np.arange(-5.12, 5.12, 0.1)
# Y = np.arange(-5.12, 5.12, 0.1)
# X, Y = np.meshgrid(X, Y)
# A = 10
# Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
# fig = plt.figure(figsize=(8,6))
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('F3(X1,X2)')
# ax.set_zlim(0,80)
# # ax.set_title('Rastrigin Function')
# plt.show()

#  rosenbrock
X = np.arange(-30, 30, 0.1)
Y = np.arange(-30, 30, 0.1)
X, Y = np.meshgrid(X, Y)
Z = 100*(Y - X * X) ** 2 + (X - 1) ** 2
fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('F4(X1,X2)')
ax.set_zlim(0,8e7)
# ax.set_title('rosenbrock Function')
plt.show()