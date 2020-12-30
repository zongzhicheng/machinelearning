from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


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


if __name__ == '__main__':
    group, labels = createDataSet()
    print(group)
    print(labels)
    print(classify0([0, 0], group, labels, 3))  # 输出：B
    print(classify0([1.0, 1.2], group, labels, 3))  # 输出：A
