# 《机器学习实战》
# 决策树
# --------------------------------
# ---------- 2020.11.06 ----------
# --------------------------------
from math import log


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # 为所有可能分类创建字典
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 以2为底求对数
    return shannonEnt


if __name__ == '__main__':
    myDat, lables = createDataSet()
    print(myDat)
    print(calcShannonEnt(myDat))
