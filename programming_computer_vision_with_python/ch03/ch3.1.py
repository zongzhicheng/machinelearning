from PIL import Image
from pylab import *
from numpy import *
from scipy.spatial import Delaunay
from scipy import linalg
from scipy import ndimage
import imageio
import os
# from xml.dom import minidom
# 就很烦，这个包找不到...
import defusedxml.minidom as minidom


def normalize(points):
    """
    在齐次坐标意义下，对点集进行归一化，使最后一行为 1
    :param points:
    :return:
    """
    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """
    将点集（dim×n的数组）转换为齐次坐标表示
    :param points:
    :return:
    """
    return vstack((points, ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    """
    使用线性DLT方法，计算单应性矩阵H，使fp映射到tp。点自动进行归一化
    :param fp:
    :param tp:
    :return:
    """
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # 对点进行归一化（对数值计算很重要）
    # --- 映射起始点 ---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = dot(C1, fp)

    # --- 映射对应点---
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = dot(C2, tp)

    # 创建用于线性方法的矩阵，对于每个对应对，在矩阵中会出现两行数值
    nbr_correspondences = fp.shape[1]
    A = zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                    tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                        tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = linalg.svd(A)
    H = V[8].reshape((3, 3))

    # 反归一化
    H = dot(linalg.inv(C2), dot(H, C1))

    # 归一化，然后返回
    return H / H[2, 2]


def Haffine_from_points(fp, tp):
    """ 计算H，仿射变换，使得tp是fp经过仿射变换H得到的"""

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # 对点进行归一化
    # --- 映射起始点---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = dot(C1, fp)

    # --- 映射对应点---
    m = mean(tp[:2], axis=1)
    C2 = C1.copy()  # 两个点集，必须都进行相同的缩放
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = dot(C2, tp)

    # 因为归一化后点的均值为0，所以平移量为0
    A = concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = linalg.svd(A.T)

    # 如Hartley和Zisserman著的Multiple View Geometry in Computer , Scond Edition 所示，
    # 创建矩阵B和C
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C, linalg.pinv(B)), zeros((2, 1))), axis=1)
    H = vstack((tmp2, [0, 0, 1]))

    # 反归一化
    H = dot(linalg.inv(C2), dot(H, C1))

    return H / H[2, 2]


def main1():
    """
    图像扭曲
    :return:
    """
    im = array(Image.open('../resource/picture/empire.jpg').convert('L'))
    H = array([[1.4, 0.05, -100], [0.05, 1.5, -100], [0, 0, 1]])
    im2 = ndimage.affine_transform(im, H[:2, :2], (H[0, 2], H[1, 2]))

    figure()
    gray()
    imshow(im2)
    show()


def image_in_image(im1, im2, tp):
    """
    使用仿射变换将im1放置在im2上，使im1图像的角和tp尽可能的靠近
    tp是齐次表示的，并且是按照从左上角逆时针计算的
    :param im1:
    :param im2:
    :param tp:
    :return:
    """
    # 扭曲的点
    m, n = im1.shape[:2]
    fp = array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    # 计算仿射变换，并且将其应用于图像im1
    H = Haffine_from_points(tp, fp)
    im1_t = ndimage.affine_transform(im1, H[:2, :2],
                                     (H[0, 2], H[1, 2]), im2.shape[:2])
    alpha = (im1_t > 0)

    return (1 - alpha) * im2 + alpha * im1_t


def main2():
    """
    仿射扭曲的一个简单例子：
    将图像或图像的一部分放置在另一幅图像中，使得它们能够和指定的区域或者标记物对齐
    :return:
    """
    # 仿射扭曲im1 到im2 的例子
    im1 = array(Image.open('../resource/picture/beatles.jpg').convert('L'))
    im2 = array(Image.open('../resource/picture/billboard_for_rent.jpg').convert('L'))

    # 选定一些目标点
    tp = array([[264, 538, 540, 264], [40, 36, 605, 605], [1, 1, 1, 1]])

    im3 = image_in_image(im1, im2, tp)

    figure()
    gray()
    imshow(im3)
    axis('equal')
    axis('off')
    show()


def alpha_for_triangle(points, m, n):
    """
    对于带有由points 定义角点的三角形，创建大小为(m，n) 的alpha 图
    （在归一化的齐次坐标意义下）
    :param points:
    :param m:
    :param n:
    :return:
    """
    alpha = zeros((m, n))
    for i in range(int(min(points[0])), int(max(points[0]))):
        for j in range(int(min(points[1])), int(max(points[1]))):
            x = linalg.solve(points, [i, j, 1])
            if min(x) > 0:  # 所有系数都大于零
                alpha[i, j] = 1
    return alpha


def simpleTriangulation():
    """
    三角剖分示例
    :return:
    """
    x, y = array(random.standard_normal((2, 100)))
    tri = Delaunay(np.c_[x, y]).simplices

    figure()
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]]  # 将第一个点加入到最后
        plot(x[t_ext], y[t_ext], 'r')

    plot(x, y, '*')
    axis('off')
    show()


def triangulate_points(x, y):
    """
     二维点的 Delaunay 三角剖分
    :param x:
    :param y:
    :return:
    """
    tri = Delaunay(np.c_[x, y]).simplices
    return tri


def pw_affine(fromim, toim, fp, tp, tri):
    """
    从一幅图像中扭曲矩形图像块
    :param fromim: 将要扭曲的图像
    :param toim: 目标图像
    :param fp: 齐次坐标表示下，扭曲前的点
    :param tp: 齐次坐标表示下，扭曲后的点
    :param tri: 三角剖分
    :return:
    """

    im = toim.copy()

    # 检查图像是灰度图像还是彩色图象
    is_color = len(fromim.shape) == 3

    # 创建扭曲后的图像（如果需要对彩色图像的每个颜色通道进行迭代操作，那么有必要这样做）
    im_t = zeros(im.shape, 'uint8')

    for t in tri:
        # 计算仿射变换
        H = Haffine_from_points(tp[:, t], fp[:, t])

        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:, :, col] = ndimage.affine_transform(
                    fromim[:, :, col], H[:2, :2], (H[0, 2], H[1, 2]), im.shape[:2])
        else:
            im_t = ndimage.affine_transform(
                fromim, H[:2, :2], (H[0, 2], H[1, 2]), im.shape[:2])

        # 三角形的alpha
        alpha = alpha_for_triangle(tp[:, t], im.shape[0], im.shape[1])

        # 将三角形加入到图像中
        im[alpha > 0] = im_t[alpha > 0]

    return im


def plot_mesh(x, y, tri):
    """
    绘制三角形
    :param x:
    :param y:
    :param tri:
    :return:
    """
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]]  # 将第一个点加入到最后
        plot(x[t_ext], y[t_ext], 'r')


# 分段仿射扭曲
def main3():
    # 打开图像，并将其扭曲
    fromim = array(Image.open('../resource/picture/sunset_tree.jpg'))
    x, y = meshgrid(range(5), range(6))
    x = (fromim.shape[1] / 4) * x.flatten()
    y = (fromim.shape[0] / 5) * y.flatten()

    # 三角剖分
    tri = triangulate_points(x, y)

    # 打开图像和目标点
    im = array(Image.open('../resource/picture/turningtorso1.jpg'))
    tp = loadtxt('../resource/picture/turningtorso1_points.txt')

    # 将点转换成齐次坐标
    fp = vstack((y, x, ones((1, len(x)))))
    tp = vstack((tp[:, 1], tp[:, 0], ones((1, len(tp)))))

    # 扭曲三角形
    im = pw_affine(fromim, im, fp, tp, tri)

    # 绘制图像
    figure()
    imshow(im)
    plot_mesh(tp[1], tp[0], tri)
    axis('off')
    show()


def read_points_from_xml(xmlFileName):
    """
    读取用于人脸对齐的控制点
    :param xmlFileName:
    :return:
    """
    xmldoc = minidom.parse(xmlFileName)
    facelist = xmldoc.getElementsByTagName('face')
    faces = {}
    for xmlFace in facelist:
        fileName = xmlFace.attributes['file'].value
        xf = int(xmlFace.attributes['xf'].value)
        yf = int(xmlFace.attributes['yf'].value)
        xs = int(xmlFace.attributes['xs'].value)
        ys = int(xmlFace.attributes['ys'].value)
        xm = int(xmlFace.attributes['xm'].value)
        ym = int(xmlFace.attributes['ym'].value)
        faces[fileName] = array([xf, yf, xs, ys, xm, ym])
    return faces


def compute_rigid_transform(refpoints, points):
    """
    计算用于将点对齐到参考点的旋转、尺度和平移量
    :param refpoints:
    :param points:
    :return:
    """
    A = array([[points[0], -points[1], 1, 0],
               [points[1], points[0], 0, 1],
               [points[2], -points[3], 1, 0],
               [points[3], points[2], 0, 1],
               [points[4], -points[5], 1, 0],
               [points[5], points[4], 0, 1]])

    y = array([refpoints[0],
               refpoints[1],
               refpoints[2],
               refpoints[3],
               refpoints[4],
               refpoints[5]])

    # 计算最小化||Ax -y || 的最小二乘解
    a, b, tx, ty = linalg.lstsq(A, y)[0]
    R = array([[a, -b], [b, a]])  # 包含尺度的旋转矩阵

    return R, tx, ty


def rigid_alignment(faces, path, plotflag=False):
    """ 严格对齐图像，并将其保存为新的图像
      path 决定对齐后图像保存的位置
      设置plotflag=True，以绘制图像"""

    # 将第一幅图像中的点作为参考点
    refpoints = list(faces.values())[0]

    # 使用仿射变换扭曲每幅图像
    for face in faces:
        points = faces[face]
        R, tx, ty = compute_rigid_transform(refpoints, points)
        T = array([[R[1][1], R[1][0]], [R[0][1], R[0][0]]])

        im = array(Image.open(os.path.join(path, face)))
        im2 = zeros(im.shape, 'uint8')

        # 对每个颜色通道进行扭曲
        for i in range(len(im.shape)):
            im2[:, :, i] = ndimage.affine_transform(im[:, :, i], linalg.inv(T), offset=[-ty, -tx])

        if plotflag:
            imshow(im2)
            show()

        # 裁剪边界，并保存对齐后的图像
        h, w = im2.shape[:2]
        border = (w + h) // 20

        # 裁剪边界
        # imsave(os.path.join(path, 'aligned/'+face),im2[border:h-border,border:w-border,:])
        # misc.imsave已经被弃用了 使用新的API接口imageio解决
        imageio.imwrite(os.path.join('aligned/' + face), im2[border:h - border, border:w - border, :])


# 图像配准
def main4():
    xmlFileName = '../resource/picture/jkfaces.xml'
    points = read_points_from_xml(xmlFileName)
    # 注册
    rigid_alignment(points, '../resource/picture/jkfaces/')


if __name__ == '__main__':
    # main1()
    # main2()
    # simpleTriangulation()
    # main3()
    main4()
