from PIL import Image
from pylab import *
from numpy import *
import os

"""
    PCV下载地址：https://github.com/jesolem/PCV
    vlfeat-0.9.20-bin.tar.gz下载地址：https://www.vlfeat.org/download/
    在anaconda环境中配置PCV和vlfeat参考博客：https://blog.csdn.net/Lv0930Hui/article/details/106254592
"""


def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """
    处理一幅图像，然后将结果保存在文件中
    :param imagename:
    :param resultname:
    :param params:
    :return:
    """
    if imagename[-3:] != 'pgm':
        # 创建一个pgm文件
        im = Image.open('../resource/picture/' + imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # 此处巨坑 要用win32 而不是win64
    cmmd = str(r"D:\vlfeat-0.9.20\bin\win32\sift.exe " + imagename + " --output=" + resultname + " " + params)
    os.system(cmmd)
    print('processd', imagename, 'to', resultname)


def read_features_from_files(filename):
    """
    读取特征属性值，然后将其以矩阵的形式返回
    :param filename:
    :return:
    """
    f = loadtxt(filename)
    return f[:, :4], f[:, 4:]  # 特征位置，描述子


def write_features_to_file(filename, locs, desc):
    """
    将特征位置和描述子保存到文件中
    :param filename:
    :param locs:
    :param desc:
    :return:
    """
    savetxt(filename, hstack((locs, desc)))


def plot_features(im, locs, circle=False):
    """
    显示带有特征的图像
    :param im: 数组图像
    :param locs: 每个特征的行、列、尺度和朝向
    :param circle:
    :return:
    """

    def draw_circle(c, r):
        t = arange(0, 1.01, .01) * 2 * pi
        x = r * cos(t) + c[0]
        y = r * sin(t) + c[1]
        plot(x, y, 'b', linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plot(locs[:, 0], locs[:, 1], 'ob')
    axis('off')


if __name__ == '__main__':
    imname = 'empire.jpg'
    im1 = array(Image.open('../resource/picture/empire.jpg').convert('L'))
    process_image(imname, 'empire.sift')
    l1, d1 = read_features_from_files('empire.sift')

    figure()
    gray()
    plot_features(im1, l1, circle=True)
    show()
