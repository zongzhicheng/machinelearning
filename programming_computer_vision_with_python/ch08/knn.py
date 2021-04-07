from numpy import *


class KnnClassifier(object):

    def __init__(self, labels, samples):
        """
        使用训练数据初始化分类器
        :param labels:
        :param samples:
        """

        self.labels = labels
        self.samples = samples

    def classify(self, point, k=3):
        """
        在训练数据上采用k近邻分类，并返回标记
        :param point:
        :param k:
        :return:
        """
        # 计算所有训练数据点的距离
        dist = array([L2dist(point, s) for s in self.samples])

        # 排序
        ndx = dist.argsort()

        # 用字典存储k近邻
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1

        return max(votes)


def L2dist(p1, p2):
    return sqrt(sum((p1 - p2) ** 2))


def L1dist(v1, v2):
    return sum(abs(v1 - v2))
