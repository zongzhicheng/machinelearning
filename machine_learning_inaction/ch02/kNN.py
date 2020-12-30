# 《机器学习实战》
# k-近邻算法
# --------------------------------
# ---------- 2020.11.03 ----------
# --------------------------------
from numpy import *
import operator


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
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


if __name__ == '__main__':
    group, labels = createDataSet()
    print(group)
    print(labels)
    print(classify0([0, 0], group, labels, 3))  # 输出：B
    print(classify0([1.0, 1.2], group, labels, 3))  # 输出：A

    # 示例：使用k-近邻算法改进约会网站的配对效果
    returnMat, classLabelVector = file2matrix('datingTestSet2.txt')
