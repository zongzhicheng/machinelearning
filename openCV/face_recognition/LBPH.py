import cv2
import numpy as np

# OpenCV提供了三种人脸识别方法，分别是LBPH、EigenFishfaces、Fisherfaces方法

# LBPH人脸识别
# LBPH（Local Binary Patterns Histogram，局部二值模式直方图）所使用的模型基于LBP （Local Binary Pattern，局部二值模式）算法
# LBP算法的基本原理是，将像素点A的值与其最邻近的8个像素点的值逐一比较：
# ● 如果A的像素值大于其临近点的像素值，则得到0。
# ● 如果A的像素值小于其临近点的像素值，则得到1。
# 最后，将像素点A与其周围8个像素点比较所得到的0、1值连起来，得到一个8位的二进制序列，将该二进制序列转换为十进制数作为点A的LBP值。
# LBP的主要思想是以当前点与其邻域像素的相对关系作为处理结果,正是因为这一点，在图像灰度整体发生变化（单调变化）时，从LBP算法中提取的特征能保持不变。

# 代码描述：已有a1、a2、a3、b1、b2、b3，检测a4属不属于a
images = []
images.append(cv2.imread("../../resources/face_picture/pg1.png", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("../../resources/face_picture/pg2.png", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("../../resources/face_picture/pg3.png", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("../../resources/face_picture/pg4.png", cv2.IMREAD_GRAYSCALE))
# 0代表是 1代表不是
labels = [0, 0, 0, 0]
# 函数cv2.face_picture.LBPHFaceRecognizer_create()语法格式为：
# retval = cv2.face_picture.LBPHFaceRecognizer_create([,radius[,neighbors[,grid_x[,gird_y[,threshold]]]]])
# ● radius：半径值，默认值为1
# ● neighbors：邻域点的个数，默认采用8邻域，根据需要可以计算更多的邻域点
# ● grid_x：将LBP特征图像划分为一个个单元格时，每个单元格在水平方向上的像素个数
# 该参数值默认为8，即将LBP特征图像在行方向上以8个像素为单位分组
# ● grid_y：将LBP特征图像划分为一个个单元格时，每个单元格在垂直方向上的像素个数
# 该参数值默认为8，即将LBP特征图像在列方向上以8个像素为单位分组
# ● threshold：在预测时所使用的阈值。如果大于该阈值，就认为没有识别到任何目标对象
recognizer = cv2.face.LBPHFaceRecognizer_create()
# 函数cv2.face_FaceRecognizer.train()对每个参考图像计算LBPH，得到一个向量。每个人脸都是整个向量集中的一个点。
# 语法格式为：
# None = cv2.face_FaceRecognizer.train(src, labels)
# ● src：训练图像，用来学习的人脸图像。
# ● labels：标签，人脸图像所对应的标签。
recognizer.train(images, np.array(labels))
predict_image = cv2.imread("../../resources/face_picture/pg5.png", cv2.IMREAD_GRAYSCALE)
# 函数cv2.face_FaceRecognizer.predict()对一个待测人脸图像进行判断，寻找与当前图像距离最近的人脸图像。
# 与哪个人脸图像最近，就将当前待测图像标注为其对应的标签。
# 如果待测图像与所有人脸图像的距离都大于函数cv2.face_picture.LBPHFaceRecognizer_create()中参数threshold所指定的距离值，则认为没有找到对应的结果，即无法识别当前人脸
# ● src：需要识别的人脸图像。
# ● label：返回的识别结果标签。
# ● confidence：返回的置信度评分。
# 置信度评分用来衡量识别结果与原有模型之间的距离。0表示完全匹配。通常情况下，认为小于50的值是可以接受的，如果该值大于80则认为差别较大。
label, confidence = recognizer.predict(predict_image)
print("label=", label)
print("confidence=", confidence)
