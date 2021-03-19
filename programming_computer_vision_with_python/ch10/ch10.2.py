import cv2
from PIL import Image
from pylab import *
from numpy import *


# 读取和写入图像
def readAndWriteImages():
    # 读取图像
    im = cv2.imread('../resource/picture/empire.jpg')
    h, w = im.shape[:2]
    print(h, w)

    # 保存图像
    cv2.imwrite('result1.jpg', im)


def colorSpace():
    im = cv2.imread('../resource/picture/empire.jpg')
    # 创建灰度图像
    bgrToGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bgrToRgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    bgrToBgra = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    imshow(bgrToGray)
    show()
    imshow(bgrToRgb)
    show()
    imshow(bgrToBgra)
    show()


def showImageAndResult1():
    # 读取图像
    im = cv2.imread('../resource/picture/fisherman.jpg')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # 计算积分图像
    intim = cv2.integral(gray)

    # 归一化并保存
    intim = (255.0 * intim) / intim.max()
    cv2.imwrite('result2.jpg', intim)


def showImageAndResult2():
    """从一个种子像素进行泛洪填充"""
    # 读取图像
    filename = '../resource/picture/fisherman.jpg'
    im = cv2.imread(filename)
    h, w = im.shape[:2]

    # 泛洪填充
    diff = (6, 6, 6)
    mask = zeros((h + 2, w + 2), uint8)
    cv2.floodFill(im, mask, (10, 10), (255, 255, 0), diff, diff)

    # 在Opencv窗口中显示结果
    cv2.imshow('flood fill', im)
    cv2.waitKey()

    # 保存结果
    cv2.imwrite('result3.jpg', im)


# SURF特征提取
def surf():
    # 读取图像
    im = cv2.imread('../resource/picture/empire.jpg')

    # 下采样
    im_lowres = cv2.pyrDown(im)

    # 变换成灰度图像
    gray = cv2.cvtColor(im_lowres, cv2.COLOR_RGB2GRAY)

    # 检测特征点
    # 用不了 会报错 module 'cv2' has no attribute 'SURF'
    # s = cv2.SURF()
    s = cv2.xfeatures2d.SURF_create()
    mask = uint8(ones(gray.shape))
    keypoints = s.detect(gray, mask)

    # 显示结果及特征点
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for k in keypoints[::10]:
        cv2.circle(vis, (int(k.pt[0]), int(k.pt[1])), 2, (0, 255, 0), -1)
        cv2.circle(vis, (int(k.pt[0]), int(k.pt[1])), int(k.size), (0, 255, 0), 2)

    cv2.imshow('local descriptors', vis)
    cv2.waitKey()


if __name__ == '__main__':
    # readAndWriteImages()
    # colorSpace()
    # showImageAndResult1()
    # showImageAndResult2()
    surf()
