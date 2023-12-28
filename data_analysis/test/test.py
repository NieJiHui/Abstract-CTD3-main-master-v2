import numpy as np
import matplotlib.pyplot as plt

# 生成一维数组
x = np.linspace(0, 1, 5)
y = np.linspace(0, 2, 3)

# 使用 np.meshgrid 创建二维网格
X, Y = np.meshgrid(x, y)

# 打印结果
print("X:")
print(X)
print("Y:")
print(Y)

# 绘制网格
plt.scatter(X, Y, marker='o')
plt.show()

