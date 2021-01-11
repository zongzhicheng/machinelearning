# 普通最小二乘法（Ordinary least squares）
# --------------------------------
# ---------- 2020.11.14 ----------
# --------------------------------

from numpy import *
import numpy as np


# 数据导入函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 标准回归函数
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 计算行列式是否为零，如果为零，那么计算逆矩阵的时候将出现错误
    # 行列式如果某两行（列）元素同比例，则行列式等于0
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # xTx.I为xTx逆矩阵
    ws = xTx.I * (xMat.T * yMat)
    return ws


xArr, yArr = loadDataSet('../resources/中央空调能耗数据.txt')
# xArr, yArr = loadDataSet('../resources/OLStest.txt')


xArr = array(xArr)
# 删除全为零的列
idx = np.argwhere(np.all(xArr[..., :] == 0, axis=0))
xArr = np.delete(xArr, idx, axis=1)
# 对xArr标准化
x_mean = xArr.mean(axis=0)
x_std = xArr.std(axis=0)
xArr = (xArr - x_mean) / x_std
# print(xArr)
# 输出xArr行数
# print(len(mat(xArr)))

# xAdd:[1,1,...,1]
xAdd = np.ones(len(mat(xArr)))
# 将xArr添加常数列
xArr = np.c_[xArr, xAdd]

ws = standRegres(xArr, yArr)
# 输出回归系数
# print(ws)

# 输出回归方程
y = "y = "
for i in range(len(ws) - 1):
    y = y + str(ws[i, 0]) + " * x" + str(i + 1) + " + "
y = y + str(ws[len(ws) - 1, 0])
print(y)

xMat = mat(xArr)
yMat = mat(yArr)
# 预测值
yHat = xMat * ws
# 相关系数
print(corrcoef(yHat.T, yMat))
