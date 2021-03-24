from PIL import Image
from pylab import *
from numpy import *


def normalize(points):
    """
    在齐次坐标意义下，对点集进行归一化，使最后一行为 1
    :param points:
    :return:
    """
    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """
    将点集（dim×n的数组）转换为齐次坐标表示
    :param points:
    :return:
    """
    return vstack((points, ones((1, points.shape[1]))))
