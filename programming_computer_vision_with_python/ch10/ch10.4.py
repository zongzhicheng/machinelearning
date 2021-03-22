import cv2
from PIL import Image
from pylab import *
from numpy import *

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
subpix_params = dict(zeroZone=(-1, -1), winSize=(10, 10),
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10)


def draw_flow(im, flow, step=16):
    """
    在间隔分开的像素采样点绘制光流
    :param im:
    :param flow:
    :param step:
    :return:
    """
    h, w = im.shape[:2]
    y, x = int32(mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1))
    # fx, fy = flow[y, x].T
    # IndexError: arrays used as indices must be of integer
    # 解决方案y,x转int
    fx, fy = flow[y, x].T

    # 创建线的终点
    lines = vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = int32(lines)

    # 创建图像并绘制
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def main1():
    # 设置视频捕获
    cap = cv2.VideoCapture(0)

    ret, im = cap.read()
    prev_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    while True:
        # 获取灰度图像
        ret, im = cap.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # 计算流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray

        # 画出流矢量
        cv2.imshow('Optical flow', draw_flow(gray, flow))
        if cv2.waitKey(10) == 27:
            break


class LKTracker(object):
    """用金字塔光流Lucas-Kanade跟踪类"""

    def __init__(self, imnames):
        """使用图像名称列表初始化"""
        self.imnames = imnames
        self.features = []
        self.tracks = []
        self.current_frame = 0

    def detect_points(self):
        """
        利用子像素精确度在当前帧中检测 "利于跟踪的好的特征（角点）"
        :param self:
        :return:
        """
        # 载入图像并创建灰度图像
        self.image = cv2.imread(self.imnames[self.current_frame])
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 搜索好的特征点
        features = cv2.goodFeaturesToTrack(self.gray, **feature_params)

        # 提炼角点位置
        cv2.cornerSubPix(self.gray, features, **subpix_params)

        self.features = features
        self.tracks = [[p] for p in features.reshape((-1, 2))]

        self.prev_gray = self.gray

    def track_points(self):
        """
        跟踪检测到的特征
        :return:
        """
        if self.features != []:
            self.step()  # 移到下一帧

            # 载入图像并创建灰度图像
            self.image = cv2.imread(self.imnames[self.current_frame])
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # reshape()操作，以适应输入格式
            tmp = float32(self.features).reshape(-1, 1, 2)

            # 计算光流
            features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, tmp, None,
                                                                     **lk_params)

            # 去除丢失的点
            self.features = [p for (st, p) in zip(status, features) if st]

            # 从丢失的点清楚跟踪轨迹
            features = array(features).reshape((-1, 2))
            for i, f in enumerate(features):
                self.tracks[i].append(f)
            ndx = [i for (i, st) in enumerate(status) if not st]
            ndx.reverse()  # 从后面移除
            for i in ndx:
                self.tracks.pop(i)

            self.prev_gray = self.gray

    def step(self, framenbr=None):
        """
        移到下一帧。如果没有给定参数，直接移到下一帧
        :param self:
        :param framenbr:
        :return:
        """
        if framenbr is None:
            self.current_frame = (self.current_frame + 1) % len(self.imnames)
        else:
            self.current_frame = framenbr % len(self.imnames)

    def draw(self):
        """
        用 OpenCV 自带的画图函数画出当前图像及跟踪点，按任意键关闭窗口
        :param self:
        :return:
        """
        # 用绿色圆圈画出跟踪点
        for point in self.features:
            cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)

        cv2.imshow('LKtrack', self.image)
        cv2.waitKey()

    def track(self):
        """
        发生器，用于遍历整个序列
        :return:
        """
        for i in range(len(self.imnames)):
            if self.features == []:
                self.detect_points()
            else:
                self.track_points()
        # 创建一个RGB副本
        f = array(self.features).reshape(-1, 2)
        im = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        yield im, f


def main2():
    imnames = ['1.pgm', '2.pgm', '3.pgm']

    # 创建跟踪对象
    lkt = LKTracker(imnames)

    # 在第一帧进行检测，跟踪剩下的帧
    lkt.detect_points()
    lkt.draw()
    for i in range(len(imnames) - 1):
        lkt.track_points()
        lkt.draw()


def main3():
    imnames = ['1.ppm', '2.ppm', '3.ppm', '4.ppm']
    # 用LKTracker发生器进行跟踪
    lkt = LKTracker(imnames)
    for im, ft in lkt.track():
        print('tracking %d features' % len(ft))

    # 画出轨迹
    figure()
    imshow(im)
    for p in ft:
        plot(p[0], p[1], 'bo')
    for t in lkt.tracks:
        plot([p[0] for p in t], [p[1] for p in t])
    axis('off')
    show()


if __name__ == '__main__':
    # main1()
    # main2()
    main3()

    # im1 = Image.open('1.jpg')
    # im2 = Image.open('2.jpg')
    # im3 = Image.open('3.jpg')
    # im4 = Image.open('4.jpg')
    #
    #
    # im1.save('1.pgm')
    # im2.save('2.pgm')
    # im3.save('3.pgm')
    # im4.save('4.pgm')
