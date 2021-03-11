from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters
from scipy.ndimage import measurements, morphology


# 灰度图像高斯模糊
def gaussianGray():
    im = array(Image.open('../resource/picture/empire.jpg').convert('L'))
    # 最后一个参数表示标准差σ，σ越大，处理后的图像细节丢失越多
    im2 = filters.gaussian_filter(im, 5)
    imshow(im)
    show()
    imshow(im2)
    show()


def gaussianColor():
    im = array(Image.open('../resource/picture/empire.jpg'))
    # 最后一个参数表示标准差σ，σ越大，处理后的图像细节丢失越多
    im2 = zeros(im.shape)
    for i in range(3):
        im2[:, :, i] = filters.gaussian_filter(im[:, :, i], 5)
    im2 = uint8(im2)

    imshow(im)
    show()
    imshow(im2)
    show()


def sobelFilters():
    im = array(Image.open('../resource/picture/empire.jpg').convert('L'))
    imshow(im)
    show()

    # Sobel滤波器
    # x的方向导数
    imx = zeros(im.shape)
    filters.sobel(im, 1, imx)
    imshow(imx)
    show()

    # y的方向导数
    imy = zeros(im.shape)
    filters.sobel(im, 0, imy)
    imshow(imy)
    show()

    # 在两个导数图像中，正导数显示为亮的像素，负导数显示为暗的像素。
    # 灰色区域表示导数的值接近于零

    # 梯度
    magnitude = sqrt(imx ** 2 + imy ** 2)
    imshow(magnitude)
    show()


def gaussianFilters():
    sigma = 5
    im = array(Image.open('../resource/picture/empire.jpg').convert('L'))
    imshow(im)
    show()

    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imshow(imx)
    show()

    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    imshow(imy)
    show()

    magnitude = sqrt(imx ** 2 + imy ** 2)
    imshow(magnitude)
    show()


# 形态学：对象计数
def morphologyExample():
    im = array(Image.open('../resource/picture/houses.png').convert('L'))
    im = 1 * (im < 128)

    labels, nbr_objects = measurements.label(im)
    print("Number of objects:", nbr_objects)


if __name__ == '__main__':
    # gaussianGray()
    # gaussianColor()
    # sobelFilters()
    # gaussianFilters()
    morphologyExample()
