# 《机器学习实战》
# AdaBoost元算法
# --------------------------------
# ---------- 2021.02.22 ----------
# --------------------------------

from numpy import *
import matplotlib.pyplot as plt


def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


# 数据可视化
def showDataSet(dataMat, labelMat):
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = array(data_plus)  # 转换为numpy矩阵
    data_minus_np = array(data_minus)  # 转换为numpy矩阵
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1])  # 正样本散点图
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1])  # 负样本散点图
    plt.show()


# 通过阙值比较对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    # 初始化retArray为1
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # 如果小于阈值,则赋值为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 如果大于阈值,则赋值为-1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    # numSteps用于在特征的所有可能值上进行遍历
    numSteps = 10.0
    # bestStump用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    # 最小误差初始化为正无穷大
    minError = inf
    # 遍历所有特征
    for i in range(n):
        # 找到特征中最小的值和最大值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # 大于和小于的情况，均遍历。lt：less than，gt：greater than
            for inequal in ['lt', 'gt']:
                # 计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 计算分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 初始化误差矩阵
                errArr = mat(ones((m, 1)))
                # 分类正确的,赋值为0
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误率
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                    i, threshVal, inequal, weightedError))
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 初始化权重
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # 构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # 计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        # 存储弱学习算法权重
        bestStump['alpha'] = alpha
        # 存储单层决策树
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        # 计算e的指数项
        # 为下一次迭代计算D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()

        # 错误率累加计算
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    showDataSet(dataMat, classLabels)
    D = mat(ones((5, 1)) / 5)
    bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)
    print(bestStump)
    print(minError)
    print(bestClasEst)
    classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
