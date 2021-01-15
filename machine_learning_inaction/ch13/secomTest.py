from numpy import *
import matplotlib.pyplot as plt
from machine_learning_inaction.ch13 import pca

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 将NaN替换成平均值的函数
def replaceNanWithMean():
    datMat = pca.loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 计算所有非NaN的平均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # 将所有NaN置为平均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


if __name__ == '__main__':
    # 利用PCA对半导体制造数据降维
    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    print(eigVals)

    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[::-1]  # 倒序
    sortedEigVals = eigVals[eigValInd]
    total = sum(sortedEigVals)
    varPercentage = sortedEigVals / total * 100

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 21), varPercentage[:20], marker='^')
    plt.xlabel('主成分数目')
    plt.ylabel('方差的百分比')
    plt.show()
