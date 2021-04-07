from PIL import Image
from numpy import *
import os


def process_image_dsift(imagename, resultname, size=20, steps=10, force_orientation=False, resize=None):
    """
    用密集采样的SIFT描述子处理一幅图像，并将结果保存在一个文件中
    :param imagename:
    :param resultname:
    :param size: 特征的大小
    :param steps: 位置之间的步长
    :param force_orientation: 是否强迫计算描述子的方位（False表示所有的方位是朝上的），用于调整图像大小的元组
    :param resize:
    :return:
    """
    im = Image.open(imagename).convert('L')
    if resize != None:
        im = im.resize(resize)
    m, n = im.size

    if imagename[-3:] != 'pgm':
        # 创建一个pgm文件
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # 创建帧，并保存到临时文件
    scale = size / 3.0
    x, y = meshgrid(range(steps, m, steps), range(steps, n, steps))
    xx, yy = x.flatten(), y.flatten()
    frame = array([xx, yy, scale * ones(xx.shape[0]), zeros(xx.shape[0])])
    savetxt('tmp.frame', frame.T, fmt='%03.3f')

    if force_orientation:
        cmmd = str("sift " + imagename + " --output=" + resultname +
                   " --read-frames=tmp.frame --orientations")
    else:
        cmmd = str("sift " + imagename + " --output=" + resultname +
                   " --read-frames=tmp.frame")
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)
