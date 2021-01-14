# 《机器学习实战》
# 主成分分析
# --------------------------------
# ---------- 2021.1.14 ----------
# --------------------------------
from numpy import *


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


# PCA算法
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    # 去平均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵 rowvar=0说明传入的数据一行代表一个样本
    # 若非0，说明传入的数据一列代表一个样本
    covMat = cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值和特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    # 从小到大对N个值排序
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    # 保留最上面的N个特征向量
    redEigVects = eigVects[:, eigValInd]
    # 将数据转换到上述N个特征向量构建的新空间中
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowMat, reconMat = pca(dataMat, 1)
