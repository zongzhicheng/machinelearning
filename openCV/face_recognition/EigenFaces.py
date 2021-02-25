import cv2
import numpy as np

# OpenCV提供了三种人脸识别方法，分别是LBPH、EigenFishfaces、Fisherfaces方法

# EigenFaces人脸识别
# EigenFaces通常也被称为特征脸，它使用主成分分析（Principal ComponentAnalysis, PCA）方法将高维的人脸数据处理为低维数据后（降维），再进行数据分析和处理，获取识别结果。

# 代码描述：已有a1、a2、a3、b1、b2、b3，检测a4属不属于a
images = []
images.append(cv2.imread("../../resources/face_picture/e1.png", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("../../resources/face_picture/e2.png", cv2.IMREAD_GRAYSCALE))

# 0代表是 1代表不是
labels = [0, 0]
# 函数cv2.face.EigenFaceRecognizer_create()的语法格式为:
# retval = cv2.face.EigenFaceRecognizer_create([, num_components[, threshold]])
# ● num_components：在PCA中要保留的分量个数。当然，该参数值通常要根据输入数据来具体确定，并没有一定之规。一般来说，80个分量就足够了。
# ● threshold：进行人脸识别时所采用的阈值。
recognizer = cv2.face.EigenFaceRecognizer_create()
# 函数cv2.face_FaceRecognizer.train()对每个参考图像进行EigenFaces计算，得到一个向量。每个人脸都是整个向量集中的一个点。
# 语法格式为：
# None = cv2.face_FaceRecognizer.train(src, labels)
# ● src：训练图像，用来学习的人脸图像
# ● labels：人脸图像所对应的标签
# PS：有个坑...
# 错误：（-210：不支持的格式或格式组合）在EigenFaces方法中，所有输入样本（训练图像）的大小必须相等！
recognizer.train(images, np.array(labels))
predict_image = cv2.imread("../../resources/face_picture/e3.png", cv2.IMREAD_GRAYSCALE)
# 函数cv2.face_FaceRecognizer.predict()在对一个待测人脸图像进行判断时，会寻找与当前图像距离最近的人脸图像。
# 与哪个人脸图像最接近，就将待测图像识别为其对应的标签
# 该函数的语法格式为：
# label, confidence = cv2.face_FaceRecognizer.predict(src)
# ● src：需要识别的人脸图像
# ● label：返回的识别结果标签
# ● confidence：返回的置信度评分。置信度评分用来衡量识别结果与原有模型之间的距离。0表示完全匹配。
# 该参数值通常在0到20000之间，只要低于5000，都被认为是相当可靠的识别结果。注意，这个范围与LBPH的置信度评分值的范围是不同的。
label, confidence = recognizer.predict(predict_image)
print("label=", label)
print("confidence=", confidence)
