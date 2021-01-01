# 《机器学习实战》
# k-近邻算法
# --------------------------------
# ---------- 2020.11.03 ----------
# --------------------------------
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 欧氏距离计算
    # tile([0,0], [4,1]) => 4行一列 [[0,0],[0,0],[0,0],[0,0]]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 沿x轴翻转dataSet
    sqDiffMat = diffMat ** 2
    # 将一个矩阵的每一行向量相加
    # np.sum([[0,1,2],[2,1,3],axis=1) => [3,6]
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 选择距离最小的k个点
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 排序
    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 详细见https://blog.csdn.net/shuiyixin/article/details/86741810
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 将文本记录转换为Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)  # 得到文件行数
    returnMat = zeros((numberOfLines, 3))  # 创建返回的Numpy矩阵
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 索引值-1表示最后一列元素
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# min-max标准化
def autoNum(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNum(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNum(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])


if __name__ == '__main__':
    group, labels = createDataSet()
    print(group)
    print(labels)
    print(classify0([0, 0], group, labels, 3))  # 输出：B
    print(classify0([1.0, 1.2], group, labels, 3))  # 输出：A

    # 示例：使用k-近邻算法改进约会网站的配对效果
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingDataMat)

    # 画散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()

    # 归一化数据
    normMat, ranges, minVals = autoNum(datingDataMat)
    print(normMat)
    print(ranges)
    print(minVals)

    datingClassTest()
    classifyPerson()
