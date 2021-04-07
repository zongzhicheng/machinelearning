import os

from PCV.localdescriptors import sift
from numpy import *

import programming_computer_vision_with_python.ch08.dsift as dsift
import programming_computer_vision_with_python.ch08.knn as knn


# 对每一辐图像创建一个特征文件
def main1():
    path1 = '../resource/picture/gesture/train/'
    path2 = '../resource/picture/gesture/test/'
    imlist = []
    for filename in os.listdir(path1):
        if os.path.splitext(filename)[1] == '.ppm':
            imlist.append(path1 + filename)
    for filename in os.listdir(path2):
        if os.path.splitext(filename)[1] == '.ppm':
            imlist.append(path2 + filename)

    # 将图像尺寸凋为(50,50)
    for filename in imlist:
        featfile = filename[:-3] + 'dsift'
        dsift.process_image_dsift(filename, featfile, 10, 5, resize=(50, 50))


def read_gesture_features_labels(path):
    # 对所有以.dsift为后缀的文件创建一个列表
    featlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dsift')]

    # 读取特征
    features = []
    for featfile in featlist:
        l, d = sift.read_features_from_file(featfile)
        features.append(d.flatten())
    features = array(features)

    # 创建标记
    labels = [featfile.split('/')[-1][0] for featfile in featlist]

    return features, array(labels)


def main2():
    # 读取训练数据
    features, labels = read_gesture_features_labels('../resource/picture/gesture/train/')
    print('training data is:', features.shape, len(labels))

    # 读取测试数据
    test_features, test_labels = read_gesture_features_labels('../resource/picture/gesture/test/')
    print('test data is:', test_features.shape, len(test_labels))

    # 可以得到一个排序后唯一的类名称列表
    classnames = unique(labels)
    print(classnames)
    nbr_classes = len(classnames)

    # 测试KNN
    k = 1
    knn_classifier = knn.KnnClassifier(labels, features)
    res = array([knn_classifier.classify(test_features[i], k)
                 for i in range(len(test_labels))])
    # 准确率
    acc = sum(1.0 * (res == test_labels)) / len(test_labels)
    print('Accuracy:', acc)
    print_confusion(res, test_labels, classnames)


def print_confusion(res, test_labels, classnames):
    n = len(classnames)

    # 混淆矩阵
    class_ind = dict([(classnames[i], i) for i in range(n)])

    confuse = zeros((n, n))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]], class_ind[test_labels[i]]] += 1

    print('Confusion matrix for')
    print(classnames)
    print(confuse)


if __name__ == '__main__':
    # main1()
    main2()
