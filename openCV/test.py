import cv2
import numpy as np


# 判断框与框重叠的情况
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


# 输入检测图像
img = cv2.imread("1.jpg")

# 创建hog算子
hog = cv2.HOGDescriptor()
# 设置svm分类器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# 基于hog算子对整个图像开展检测
found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.05)
# 对检测到的图像box进行重合去除处理
found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_filtered.append(r)

# 对box中的行人进行标注
for person in found_filtered:
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

# 显示检测图像
cv2.imshow("people detection", img)
cv2.waitKey(0)
