from PIL import Image
from pylab import *
from numpy import *
from scipy import linalg
from PCV.localdescriptors import sift
import programming_computer_vision_with_python.ch04.homography as homography
import pickle


class Camera(object):
    """表示针孔照相机的类"""

    def __init__(self, P):
        """
        初始化P = K[R|t]照相机模型
        :param P:
        """
        self.P = P
        self.K = None  # calibration matrix
        self.R = None  # rotation
        self.t = None  # translation
        self.c = None  # camera center

    def project(self, X):
        """
        X（4xn的数组）的投影点，，并且进行坐标归一化
        :param X:
        :return:
        """
        x = dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
        return x

    def factor(self):
        """
        将照相机矩阵分解为K、R、t，其中，P = K[R|t]
        :return:
        """
        # 分解前3x3的部分
        K, R = linalg.rq(self.P[:, :3])

        # 将K的对角线元素设为正值
        T = diag(sign(diag(K)))
        if linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = dot(K, T)
        self.R = dot(T, R)  # T的逆矩阵为其自身
        self.t = dot(linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t

    def center(self):
        """
        计算并返回照相机的中心
        :return:
        """
        if self.c is not None:
            return self.c
        else:
            # 通过因式分解计算c
            self.factor()
            self.c = -dot(self.R.T, self.t)
            return self.c


def rotation_matrix(a):
    """
    创建一个用于围绕向量a轴旋转的三维旋转矩阵
    :param a:
    :return:
    """
    R = eye(4)
    R[:3, :3] = linalg.expm([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return R


def my_calibration(sz):
    """
    测量数据辅助函数
    :param sz: 表示图像大小的元组
    :return: 返回参数为标定矩阵
    """
    row, col = sz
    fx = 2555 * col / 2592
    fy = 2586 * row / 1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5 * col
    K[1, 2] = 0.5 * row
    return K


def main1():
    # 原书本下载地址已经失效 新地址：https://www.robots.ox.ac.uk/~vgg/data/mview/
    # 载入点
    points = loadtxt('../resource/picture/3D/house.p3d').T
    points = vstack((points, ones(points.shape[1])))

    # 设置照相机参数
    P = hstack((eye(3), array([[0], [0], [-10]])))
    cam = Camera(P)
    x = cam.project(points)

    # 绘制投影
    figure()
    plot(x[0], x[1], 'k.')
    show()

    # 创建变换
    r = 0.05 * random.rand(3)
    rot = rotation_matrix(r)

    # 旋转矩阵和投影
    figure()
    for t in range(20):
        cam.P = dot(cam.P, rot)
        x = cam.project(points)
        plot(x[0], x[1], 'k.')
    show()


def main2():
    K = array([[100, 0, 500], [0, 1000, 300], [0, 0, 1]])
    tmp = rotation_matrix([0, 0, 1])[:3, :3]
    Rt = hstack((tmp, array([[50], [40], [30]])))
    cam = Camera(dot(K, Rt))

    # 得到相同的输出
    print(K, Rt)
    print(cam.factor())


def cube_points(c, wid):
    """
    创建用于绘制立方体的一个点列表（前5个点是底部的正方形，一些边重合了）
    :param c:
    :param wid:
    :return:
    """
    p = []
    # 底部
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])  # 为了绘制闭合图像，和第一个相同

    # 顶部
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])  # 为了绘制闭合图像，和第一个相同

    # 竖直边
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])

    return array(p).T


def main3():
    sift.process_image('../resource/picture/book_frontal.JPG', 'im0.sift')
    l0, d0 = sift.read_features_from_file('im0.sift')

    sift.process_image('../resource/picture/book_perspective.JPG', 'im1.sift')
    l1, d1 = sift.read_features_from_file('im1.sift')

    # 匹配特征，并计算单应性矩阵
    matches = sift.match_twosided(d0, d1)
    ndx = matches.nonzero()[0]
    fp = homography.make_homog(l0[ndx, :2].T)
    ndx2 = [int(matches[i]) for i in ndx]
    tp = homography.make_homog(l1[ndx2, :2].T)

    model = homography.RansacModel()
    H, inliers = homography.H_from_ransac(fp, tp, model)

    # 计算照相机标定矩阵
    K = my_calibration((747, 1000))

    # 位于边长为0.2，z=0平面上的三维点
    box = cube_points([0, 0, 0.1], 0.1)

    # 投影第一幅图像上底部的正方形
    cam1 = Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
    # 底部正方形上的点
    box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

    # 使用H将点变换到第二幅图像中
    box_trans = homography.normalize(dot(H, box_cam1))

    # 从cam1和H中计算第二个照相机矩阵
    cam2 = Camera(dot(H, cam1.P))
    A = dot(linalg.inv(K), cam2.P[:, :3])
    A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
    cam2.P[:, :3] = dot(K, A)

    # 使用第二个照相机矩阵投影
    box_cam2 = cam2.project(homography.make_homog(box))

    # 测试：将点投影在z=0上，应该能够得到相同的点
    point = array([1, 1, 0, 1]).T
    print(homography.normalize(dot(dot(H, cam1.P), point)))
    print(cam2.project(point))

    # 可视化
    im0 = array(Image.open('../resource/picture/book_frontal.JPG'))
    im1 = array(Image.open('../resource/picture/book_perspective.JPG'))

    # 底部正方形的二维投影
    figure()
    imshow(im0)
    plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)
    title('2D projection of bottom square')
    axis('off')

    # 使用H对二维投影进行变换
    figure()
    imshow(im1)
    plot(box_trans[0, :], box_trans[1, :], linewidth=3)
    title('2D projection transfered with H')
    axis('off')

    # 三维立方体
    figure()
    imshow(im1)
    plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)
    title('3D points projected in second image')
    axis('off')

    show()

    with open('ar_camera.pkl', 'wb') as f:
        pickle.dumps(K, 3)
        pickle.dump(dot(linalg.inv(K), cam2.P), f)


if __name__ == '__main__':
    # main1()
    # main2()
    main3()
