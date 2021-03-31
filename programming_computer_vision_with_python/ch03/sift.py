from PIL import Image
from pylab import *
from numpy import *
import os

"""
实际上和ch2.2.py里的方法是一样的 但是调用比较麻烦，复制一份
"""


def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """
    处理一幅图像，然后将结果保存在文件中
    :param imagename:
    :param resultname:
    :param params:
    :return:
    """
    if imagename[-3:] != 'pgm':
        # 创建一个pgm文件
        im = Image.open('../resource/picture/' + imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # 现在把sift.exe直接加在项目里 这样就可以直接使用了
    cmmd = str(r"sift.exe " + imagename + " --output=" + resultname + " " + params)

    # 此处巨坑 要用win32 而不是win64
    # cmmd = str(r"D:\vlfeat-0.9.20\bin\win32\sift.exe " + imagename + " --output=" + resultname + " " + params)
    os.system(cmmd)
    print('processd', imagename, 'to', resultname)


def read_features_from_file(filename):
    """
    读取特征属性值，然后将其以矩阵的形式返回
    :param filename:
    :return:
    """
    f = loadtxt(filename)
    return f[:, :4], f[:, 4:]  # 特征位置，描述子


def match(desc1, desc2):
    """
    对于第一幅图像中的每个描述子，选取其在第二幅图像中的匹配
    :param desc1: 第一幅图像的描述子
    :param desc2: 第二幅图象的描述子
    :return:
    """
    desc1 = array([d / linalg.norm(d) for d in desc1])
    desc2 = array([d / linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = zeros((desc1_size[0], 1), 'int')
    # 预先计算矩阵转置
    desc2t = desc2.T
    for i in range(desc1_size[0]):
        # 向量点乘
        dotprods = dot(desc1[i, :], desc2t)
        dotprods = 0.9999 * dotprods
        # 反余弦和反排序，返回第二幅图像中特征的索引
        indx = argsort(arccos(dotprods))

        # 检查最近邻的角度是否小于dist_ratio乘以第二近邻的角度
        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores
