# Numpy学习源码
# 创建时间2020.10.04
# 《Python深度学习：基于PyTorch》
# --------------------------------
# ---------- 2020.10.04 ----------
# --------------------------------
import numpy as np
from numpy import random as nr

# 将列表转换成ndarray
list1 = [3.14, 2.17, 0, 1, 2]
nd1 = np.array(list1)
print(nd1)
print(type(nd1))

# 嵌套列表可以转换成多维ndarray
list2 = [[3.14, 2.17, 0, 1, 2], [1, 2, 3, 4, 5]]
nd2 = np.array(list2)
print(nd2)

# 利用random模块生成数组
# np.random.random 生成0到1之间的随机数
# np.random.uniform 生成均匀分布的随机数
# np.random.rand 生成标准正态的随机数
# np.random.normal 生成正态分布
# np.random.shuffle 随机打乱顺序
# np.random.seed 设置随机数种子
nd3 = np.random.random([3, 3])
print(nd3)
print("nd3的形状为", nd3.shape)

np.random.seed(123)
nd4 = np.random.randn(2, 3)
print(nd4)
np.random.shuffle(nd4)
print(nd4)
print(type(nd4))

# 创建特定形状的多维数组
# np.zeros((3, 4)) 创建3x4的元素全为0的数组
# np.ones((3, 4)) 创建3x4的元素全为1的数组
# np.empty((3, 4)) 创建3x4的的空数组，空数据中的值并不为0，而是未初始化的垃圾值
# np.zeros_like(ndarr) 以ndarr相同维度创建元素全为0数组
# np.eye(5) 创建一个5x5的矩阵，对角线为1，其余为0
nd5 = np.zeros([3, 3])
nd6 = np.ones([3, 3])
nd7 = np.eye(3)
nd8 = np.diag([1, 2, 3])  # 生成3阶对角矩阵
print(nd5)
print(nd6)
print(nd7)
print(nd8)
# 将生产的数据暂时保存起来，以备后续使用
nd9 = np.random.random([5, 5])
np.savetxt(X=nd9, fname='./test1.txt')
nd10 = np.loadtxt('./test1.txt')
print(nd10)

# 获取元素
np.random.seed(2019)
nd11 = np.random.random([10])
print(nd11)
# 获取指定位置的数据，获取第4个元素
print(nd11[3])
# 截取一段数据
print(nd11[3:6])
# 截取固定间隔数据 例如：1到6 间隔2
print(nd11[1:6:2])
# 倒序取数
print(nd11[::-1])

a = np.arange(1, 25, dtype=float)
c1 = nr.choice(a, size=(3, 4))  # size指定输出数组形状
c2 = nr.choice(a, size=(3, 4), replace=False)  # replace缺省为True，即可重复抽取
# 下式中参数p指定每个元素对应的抽取概率，缺省为每个元素被抽取的概率相同
c3 = nr.choice(a, size=(3, 4), p=a / np.sum(a))
print("随机可重复抽取", c1)
print("随机但不重复抽取", c2)
print("随机但按制度概率抽取", c3)

# 矩阵对应元素相乘
A = np.array([[1, 2], [-1, 4]])
B = np.array([[2, 0], [3, 4]])
print(A * B)
np.multiply(A, B)
# 点积运算
x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([[5, 6, 7], [8, 9, 10]])
print(np.dot(x1, x2))

# arr.reshape 重新将向量arr维度进行改变，不修改向量本身
# arr.resize 重新将向量arr维度进行改变，修改向量本身
# arr.T 对向量arr进行转置
# arr.transpose 对高维矩阵进行轴对换
arr = np.arange(10)
print(arr)
print(arr.reshape(2, 5))
arr.resize(2, 5)
print(arr)
print(arr.T)
arr2 = np.arange(24).reshape(2, 3, 4)
print(arr2.shape)
print(arr2.transpose(1, 2, 0).shape)

# 生成10000个形状为2x3的矩阵
data_train = np.random.randn(10000, 2, 3)
print(data_train.shape)
# 打乱这10000条数据
np.random.shuffle(data_train)
# 定义批量大小
batch_size = 1000
# 进行批处理
for i in range(0, len(data_train), batch_size):
    x_batch_sum = np.sum(data_train[i:i + batch_size])
    print("第{}批次，该批次的数据之和：{}".format(i, x_batch_sum))

