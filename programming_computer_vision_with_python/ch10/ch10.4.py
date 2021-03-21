import cv2
from PIL import Image
from pylab import *
from numpy import *


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


def main2():
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    subpix_params = dict(zeroZone=(-1, -1), winSize=(10, 10),
                         criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10)

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


if __name__ == '__main__':
    main1()
