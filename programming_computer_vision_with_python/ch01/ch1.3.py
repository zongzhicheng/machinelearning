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


if __name__ == '__main__':
    # example1()
    # example2()

    im = array(Image.open('../resource/picture/AquaTermi_lowcontrast.jpg').convert('L'))
    im2, cdf = histeq(im)
    hist(im2.flatten(), 128)
    show()
    imshow(im2)
    show()
