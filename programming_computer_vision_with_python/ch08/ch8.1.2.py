import programming_computer_vision_with_python.ch08.dsift as dsift
from PIL import Image
from pylab import *


def read_features_from_files(filename):
    """
    读取特征属性值，然后将其以矩阵的形式返回
    :param filename:
    :return:
    """
    f = loadtxt(filename)
    return f[:, :4], f[:, 4:]  # 特征位置，描述子


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


dsift.process_image_dsift('../resource/picture/empire.jpg', 'empire.sift', 90, 40, True)
l, d = read_features_from_files('empire.sift')

im = array(Image.open('../resource/picture/empire.jpg'))
plot_features(im, l, True)
show()
