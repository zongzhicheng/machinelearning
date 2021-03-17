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

    # 现在把sift.exe直接加在项目里 这样就可以直接使用了
    cmmd = str(r"sift.exe " + imagename + " --output=" + resultname + " " + params)

    # 此处巨坑 要用win32 而不是win64
    # cmmd = str(r"D:\vlfeat-0.9.20\bin\win32\sift.exe " + imagename + " --output=" + resultname + " " + params)
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


def main1():
    """
    使用圆圈表示特征尺度的SIFT特征
    :return:
    """
    imname = 'empire.jpg'
    im1 = array(Image.open('../resource/picture/empire.jpg').convert('L'))
    process_image(imname, 'empire.sift')
    l1, d1 = read_features_from_files('empire.sift')

    figure()
    gray()
    plot_features(im1, l1, circle=True)
    show()


def match(desc1, desc2):
    """
    对于第一幅图像中的每个描述子，选取其在第二幅图像中的匹配
    :param desc1: 第一幅图像的描述子
    :param desc2: 第二幅图象的描述子
    :return:
    """
    desc1 = array([d / linalg.norm(d) for d in desc1])
    desc2 = array([d / linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = zeros((desc1_size[0], 1), 'int')
    # 预先计算矩阵转置
    desc2t = desc2.T
    for i in range(desc1_size[0]):
        # 向量点乘
        dotprods = dot(desc1[i, :], desc2t)
        dotprods = 0.9999 * dotprods
        # 反余弦和反排序，返回第二幅图像中特征的索引
        indx = argsort(arccos(dotprods))

        # 检查最近邻的角度是否小于dist_ratio乘以第二近邻的角度
        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores


def match_twosided(desc1, desc2):
    """
    双向对称版本的match()
    :param desc1:
    :param desc2:
    :return:
    """
    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # 去除非对称的匹配
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12


def appendimages(im1, im2):
    """
    返回将两幅图像并排拼接成的一幅新图像
    :param im1:
    :param im2:
    :return:
    """
    # 选取具有最少行数的图像，然后填充足够的空行
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im1, zeros((rows1 - rows2, im1.shape[1]))), axis=0)
    # 如果这些情况都没有，那么它们的行数相同，不需要进行填充

    return concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """
    显示一幅带有连接匹配之间连线的图片
    :param im1: 数组图像
    :param im2: 数组图像
    :param locs1: 特征位置
    :param locs2: 特征位置
    :param matchscores: match()的输出
    :param show_below: 如果图像应该显示在匹配的下方
    :return:
    """
    im3 = appendimages(im1, im2)
    if show_below:
        im3 = vstack((im3, im3))

    imshow(im3)
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m[0] > 0:
            plot([locs1[i][1], locs2[m[0]][1] + cols1], [locs1[i][0], locs2[m[0]][0]], 'c')
    axis('off')


def main2():
    """
    在两幅图像间匹配SIFT特征
    :return:
    """
    im1 = array(Image.open('../resource/picture/climbing_1_small.jpg').convert('L'))
    im2 = array(Image.open('../resource/picture/climbing_2_small.jpg').convert('L'))

    process_image('climbing_1_small.jpg', 'climbing_1_small.sift')
    process_image('climbing_2_small.jpg', 'climbing_2_small.sift')

    l1, d1 = read_features_from_files('climbing_1_small.sift')
    l2, d2 = read_features_from_files('climbing_2_small.sift')

    figure()
    gray()
    plot_features(im1, l1, circle=True)
    show()

    figure()
    gray()
    plot_features(im2, l2, circle=True)
    show()

    print("starting matching")
    startTime = time.time()
    matches = match(d1, d2)
    matches_twosided = match_twosided(d1, d2)
    endTime = time.time()
    print("计算时间：" + str(endTime - startTime))

    figure()
    gray()
    plot_matches(im1, im2, l1, l2, matches, show_below=True)
    show()

    # 使用对称匹配条件可以去除不正确的匹配，保留好的匹配（一些正确的匹配也被去除了）
    figure()
    gray()
    plot_matches(im1, im2, l1, l2, matches_twosided, show_below=True)
    show()


if __name__ == '__main__':
    # main1()

    main2()
