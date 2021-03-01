from PIL import Image
from pylab import *


def example1():
    im = array(Image.open('../resource/picture/empire.jpg'))
    # (800, 569, 3) -> (行, 列, 颜色通道)
    print(im.shape, im.dtype)

    im = array(Image.open('../resource/picture/empire.jpg').convert('L'), 'f')
    print(im.shape, im.dtype)


def example2():
    im = array(Image.open('../resource/picture/empire.jpg').convert('L'))
    imshow(im)
    show()
    print(int(im.min()), int(im.max()))
    # 对图像进行反相处理
    im2 = 255 - im
    imshow(im2)
    show()
    print(int(im2.min()), int(im2.max()))
    # 将图像像素值变换到[100,200]
    im3 = (100.0 / 255) * im + 100
    imshow(im3)
    show()
    print(int(im3.min()), int(im3.max()))
    # 对图像像素值求p平方后得到的图像
    im4 = 255.0 * (im / 255.0) ** 2
    imshow(im4)
    show()
    print(int(im4.min()), int(im4.max()))


# 对一幅灰度图像进行直方图均衡化
def histeq(im, nbr_bins=256):
    # 绘制原始灰度图像的直方图
    hist(im.flatten(), 128)
    show()
    # 绘制原始灰度图像
    imshow(im)
    show()

    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    # 累积分布函数
    cdf = imhist.cumsum()
    # 归一化
    cdf = 255 * cdf / cdf[-1]

    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


# 计算图像列表的平均图像
def compute_average(imlist):
    # 打开第一幅图像，将其存储在浮点型数组中
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + '...skipped')
    averageim /= len(imlist)
    # 返回uint8类型的平均图像
    return array(averageim, 'uint8')


# 主成分分析
def pca(X):
    """
    :param X:矩阵X中存储训练数据，每一行为一条训练数据
    :return:投影矩阵（按照维度的重要性排序）、方差和均值
    """
    # 获取维数
    num_data, dim = X.shape

    # 数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA-使用紧致技巧
        # 协方差矩阵
        M = dot(X, X.T)
        # 特征值和特征向量
        e, EV = linalg.eigh(M)
        # 这就是紧致技巧
        tmp = dot(X.T, EV).T
        # 由于最后的特征向量是我们所需要的，所以需要将其逆转
        V = tmp[::-1]
        # 由于特征值是按照递增顺序排列的，所以需要将其逆转
        S = sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA-使用SVD方法
        U, S, V = linalg.svd(X)
        # 仅仅返回当前num_data维的数据才合理
        V = V[:num_data]
    # 返回投影矩阵、方差和均值
    return V, S, mean_X


if __name__ == '__main__':
    # example1()
    # example2()

    im = array(Image.open('../resource/picture/AquaTermi_lowcontrast.jpg').convert('L'))
    im2, cdf = histeq(im)
    hist(im2.flatten(), 128)
    show()
    imshow(im2)
    show()
