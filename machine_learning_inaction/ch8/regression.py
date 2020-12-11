# 《机器学习实战》
# 多元线性回归
# --------------------------------
# ---------- 2020.11.03 ----------
# --------------------------------
from numpy import *
import matplotlib.pyplot as plt


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
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # xTx.I为xTx逆矩阵
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部加权线性回归函数
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建对角矩阵
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print
        "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


xArr, yArr = loadDataSet('ex0.txt')

ws = standRegres(xArr, yArr)
print(ws)

xMat = mat(xArr)
yMat = mat(yArr)
# 预测值
yHat = xMat * ws

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()

yHat = xMat * ws
# 相关系数
print(corrcoef(yHat.T, yMat))

# lwlr(xArr[0], xArr, yArr, 1.0)
