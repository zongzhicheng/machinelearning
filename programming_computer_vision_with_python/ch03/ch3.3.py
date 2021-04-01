import programming_computer_vision_with_python.ch03.sift as sift
from PIL import Image
from pylab import *
from numpy import *
from scipy import ndimage


# 使用SIFT特征自动找到匹配对应
def main1():
    featname = ['Univ' + str(i + 1) + '.sift' for i in range(5)]
    imname = ['Univ' + str(i + 1) + '.jpg' for i in range(5)]
    l = {}
    d = {}
    for i in range(5):
        sift.process_image(imname[i], featname[i])
        l[i], d[i] = sift.read_features_from_file(featname[i])

    matches = {}
    for i in range(4):
        matches[i] = sift.match(d[i + 1], d[i])
    return l, matches


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


class RansacModel(object):
    """
    用于测试单应性矩阵的类，其中单应性矩阵是由网站http://www.scipy.org/Cookbook/RANSAC上的ransac.py计算出来的
    """

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """计算选取的4个对应的单应性矩阵"""

        # 将其转置，来调用H_from_points() 计算单应性矩阵
        data = data.T

        # 映射的起始点
        fp = data[:3, :4]
        # 映射的目标点
        tp = data[3:, :4]

        # 计算单应性矩阵，然后返回
        return H_from_points(fp, tp)

    def get_error(self, data, H):
        """对所有的对应计算单应性矩阵，然后对每个变换后的点，返回相应的误差"""

        data = data.T

        # 映射的起始点
        fp = data[:3]
        # 映射的目标点
        tp = data[3:]

        # 变换fp
        fp_transformed = dot(H, fp)

        # 归一化齐次坐标
        for i in range(3):
            fp_transformed[i] /= fp_transformed[2]

        # 返回每个点的误差
        return sqrt(sum((tp - fp_transformed) ** 2, axis=0))


def H_from_ransac(fp, tp, model, maxiter=1000, match_theshold=10):
    """
    使用RANSAC 稳健性估计点对应间的单应性矩阵
    （ransac.py 为从 http://www.scipy.org/Cookbook/RANSAC 下载的版本
    已下载至当前目录下）
    :param fp: 3×n 的数组
    :param tp: 3×n 的数组
    :param model:
    :param maxiter:
    :param match_theshold:
    :return:
    """
    import programming_computer_vision_with_python.ch03.ransac as ransac
    # 对应点组
    data = vstack((fp, tp))

    # 计算H，并返回
    H, ransac_data = ransac.ransac(data.T, model, 4, maxiter, match_theshold, 10, return_all=True)
    return H, ransac_data['inliers']


# 将匹配转换成齐次坐标点的函数
def convert_points(j):
    l, matches = main1()
    ndx = matches[j].nonzero()[0]
    fp = make_homog(l[j + 1][ndx, :2].T)
    ndx2 = [int(matches[j][i]) for i in ndx]
    tp = make_homog(l[j][ndx2, :2].T)
    return fp, tp


def main2():
    # 估计单应性矩阵
    model = RansacModel()

    fp, tp = convert_points(1)
    H_12 = H_from_ransac(fp, tp, model)[0]  # im1 到im2 的单应性矩阵

    fp, tp = convert_points(0)
    H_01 = H_from_ransac(fp, tp, model)[0]  # im0 到im1 的单应性矩阵

    tp, fp = convert_points(2)  # 注意：点是反序的
    H_32 = H_from_ransac(fp, tp, model)[0]  # im3 到im2 的单应性矩阵

    tp, fp = convert_points(3)  # 注意：点是反序的
    H_43 = H_from_ransac(fp, tp, model)[0]  # im4 到im3 的单应性矩阵
    return H_12, H_01, H_32, H_43


def panorama(H, fromim, toim, padding=2400, delta=2400):
    """ 使用单应性矩阵H（使用RANSAC 健壮性估计得出），协调两幅图像，创建水平全景图像。结果
      为一幅和toim 具有相同高度的图像。padding 指定填充像素的数目，delta 指定额外的平移量"""

    # 检查图像是灰度图像，还是彩色图像
    is_color = len(fromim.shape) == 3

    # 用于geometric_transform() 的单应性变换
    def transf(p):
        p2 = dot(H, [p[0], p[1], 1])
        return p2[0] / p2[2], p2[1] / p2[2]

    if H[1, 2] < 0:  # fromim 在右边
        print('warp - right')
        # 变换fromim
        if is_color:
            # 在目标图像的右边填充0
            toim_t = hstack((toim, zeros((toim.shape[0], padding, 3))))
            fromim_t = zeros((toim.shape[0], toim.shape[1] + padding, toim.shape[2]))
            for col in range(3):
                fromim_t[:, :, col] = ndimage.geometric_transform(fromim[:, :, col],
                                                                  transf, (toim.shape[0], toim.shape[1] + padding))
        else:
            # 在目标图像的右边填充0
            toim_t = hstack((toim, zeros((toim.shape[0], padding))))
            fromim_t = ndimage.geometric_transform(fromim, transf,
                                                   (toim.shape[0], toim.shape[1] + padding))
    else:
        print('warp - left')
        # 为了补偿填充效果，在左边加入平移量
        H_delta = array([[1, 0, 0], [0, 1, -delta], [0, 0, 1]])
        H = dot(H, H_delta)
        # fromim 变换
        if is_color:
            # 在目标图像的左边填充0
            toim_t = hstack((zeros((toim.shape[0], padding, 3)), toim))
            fromim_t = zeros((toim.shape[0], toim.shape[1] + padding, toim.shape[2]))
            for col in range(3):
                fromim_t[:, :, col] = ndimage.geometric_transform(fromim[:, :, col],
                                                                  transf, (toim.shape[0], toim.shape[1] + padding))

        else:
            # 在目标图像的左边填充0
            toim_t = hstack((zeros((toim.shape[0], padding)), toim))
            fromim_t = ndimage.geometric_transform(fromim,
                                                   transf, (toim.shape[0], toim.shape[1] + padding))

    # 协调后返回（将fromim 放置在toim 上）
    if is_color:
        # 所有非黑色像素
        alpha = ((fromim_t[:, :, 0] * fromim_t[:, :, 1] * fromim_t[:, :, 2]) > 0)
        for col in range(3):
            toim_t[:, :, col] = fromim_t[:, :, col] * alpha + toim_t[:, :, col] * (1 - alpha)
    else:
        alpha = (fromim_t > 0)
        toim_t = fromim_t * alpha + toim_t * (1 - alpha)

    return toim_t


def main3():
    # 扭曲图像
    delta = 2000  # 用于填充和平移
    imname = ['../resource/picture/Univ' + str(i + 1) + '.jpg' for i in range(5)]
    H_12, H_01, H_32, H_43 = main2()
    im1 = array(Image.open(imname[1]), "uint8")
    im2 = array(Image.open(imname[2]), "uint8")
    im_12 = panorama(H_12, im1, im2, delta, delta)

    figure()
    imshow(array(im_12, 'uint8'))
    axis('off')
    show()

    im1 = array(Image.open(imname[0]), "f")
    im_02 = panorama(dot(H_12, H_01), im1, im_12, delta, delta)

    figure()
    imshow(array(im_02, 'uint8'))
    axis('off')
    show()

    im1 = array(Image.open(imname[3]), "f")
    im_32 = panorama(H_32, im1, im_02, delta, delta)

    figure()
    imshow(array(im_32, 'uint8'))
    axis('off')
    show()

    im1 = array(Image.open(imname[4]), "f")
    im_42 = panorama(dot(H_32, H_43), im1, im_32, delta, 2 * delta)

    figure()
    imshow(array(im_42, 'uint8'))
    axis('off')
    show()


if __name__ == '__main__':
    # l, matches = main1()
    # main2()
    main3()
    print("end")
