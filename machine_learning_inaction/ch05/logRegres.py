# 《机器学习实战》
# Logistic回归
# --------------------------------
# ---------- 2020.11.03 ----------
# --------------------------------
from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid函数
def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))


# Logistic回归梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 转换为Numpy矩阵数据类型
    labelMat = mat(classLabels).transpose()  # 转置函数
    m, n = shape(dataMatrix)
    alpha = 0.001  # alpha是向目标移动的步长
    maxCycles = 500  # maxCycles是迭代次数
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 画出Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # 初始化每个回归系数为1
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # 初始化每个回归系数为1
    for j in range(numIter):
        # 书中代码原本是dataIndex = range(m)
        # python3.x中 range返回的是range对象，不是数组对象
        # 所以改为list
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha每次迭代时需要调整
            # 随机选取更新
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # 梯度上升算法
    # weights = gradAscent(dataArr, labelMat)
    # print(weights)
    # plotBestFit(weights.getA())

    # 随机梯度上升算法
    # weights = stocGradAscent0(array(dataArr), labelMat)
    # print(weights)
    # plotBestFit(weights)

    # 改进的随机梯度上升算法
    weights = stocGradAscent1(array(dataArr), labelMat)
    print(weights)
    plotBestFit(weights)
