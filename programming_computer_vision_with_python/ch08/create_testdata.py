from numpy.random import randn
import pickle
from numpy import *

# 创建二维样本数据
n = 200

# 两个正态分布数据集
class_1 = 0.6 * randn(n, 2)
class_2 = 1.2 * randn(n, 2) + array([5, 1])
labels = hstack((ones(n), -ones(n)))

# 用pickle模块保存
with open('points_normal_test.pkl', 'wb') as f:
    pickle.dump(class_1, f)
    pickle.dump(class_2, f)
    pickle.dump(labels, f)

# 正态分布，并使数据成环绕状分布
class_1 = 0.6 * randn(n, 2)
r = 0.8 * randn(n, 1) + 5
angle = 2 * pi * randn(n, 1)
class_2 = hstack((r * cos(angle), r * sin(angle)))
labels = hstack((ones(n), -ones(n)))

# 用pickle保存
with open('points_ring_test.pkl', 'wb') as f:
    pickle.dump(class_1, f)
    pickle.dump(class_2, f)
    pickle.dump(labels, f)
