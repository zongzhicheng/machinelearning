import cv2
import numpy as np

# 代码描述：使用cv2.VideoCapture类捕获摄像头视频。
# 捕获对象 = cv2.VideoCapture("摄像头ID号")
# 默认值为-1，表示随机选择一个摄像头
# 如果有多个摄像头，则用数字"0"表示第1个摄像头，用数字"1"表示第2个摄像头，以此类推
cap = cv2.VideoCapture(0)
# cv2.VideoCapture.isOpened检查初始化是否成功
# 语法规则为：
# retval = cv2.VideoCapture.isOpened(index)
# ● index为摄像头ID号
# ● retval为返回值，当摄像头（或者视频文件）被成功打开时，返回值为True。
while (cap.isOpened()):
    # 捕获帧,该函数语法规则：
    # retval, image = cv2.VideoCapture.read()
    # ● image是返回的捕获到的帧，如果没有帧被捕获，则该值为空。
    # ● retval表示捕获是否成功，如果成功则该值为True，不成功则为False。
    ret, frame = cap.read()
    # 将这帧转换为灰度图
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)
    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    # ESC键
    if c == 27:
        break
# 释放
cap.release()
cv2.destroyAllWindows()
