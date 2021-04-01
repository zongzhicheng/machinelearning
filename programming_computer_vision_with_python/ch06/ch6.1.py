from PIL import Image, ImageDraw
from pylab import *
from numpy import *
from PCV.tools import imtools, pca
from PCV.clustering import hcluster
import pickle
from scipy.cluster.vq import *
import cv2


def main1():
    """
    SciPy聚类包 k-means聚类
    :return:
    """
    class1 = 1.5 * randn(100, 2)
    class2 = randn(100, 2) + array([5, 5])
    features = vstack((class1, class2))
    centroids, variance = kmeans(features, 2)
    # 对每个数据点进行归类
    code, distance = vq(features, centroids)
    figure()
    ndx = where(code == 0)[0]
    plot(features[ndx, 0], features[ndx, 1], '*')
    ndx = where(code == 1)[0]
    plot(features[ndx, 0], features[ndx, 1], 'r.')
    plot(centroids[:, 0], centroids[:, 1], 'go')
    axis('off')
    show()


def main2():
    """
    图像聚类
    :return:
    """
    # 获取selected-fontimages文件下的图像文件名，并保存在列表中
    imlist = imtools.get_imlist('../resource/picture/selectedfontimages/a_selected_thumbs/')
    imnbr = len(imlist)
    print("The number of images is %d" % imnbr)

    # 载入模型文件
    with open('../resource/picture/selectedfontimages/a_pca_modes.pkl', 'rb') as f:
        # 在python3中，sys.setdefaultencoding(‘utf-8’)已被禁用，
        # 将导入文件代码加上encoding='bytes'则可解决：
        immean = pickle.load(f, encoding='bytes')
        V = pickle.load(f, encoding='bytes')

    # 创建矩阵，存储所有拉成一组形式后的图像
    immatrix = array([array(Image.open(imname)).flatten() for imname in imlist], 'f')

    # 投影到40个主成分上
    immean = immean.flatten()
    projected = array([dot(V[:40], immatrix[i] - immean) for i in range(imnbr)])

    # 进行k-means聚类
    projected = whiten(projected)
    centroids, distortion = kmeans(projected, 4)
    code, distance = vq(projected, centroids)

    # 绘制聚类簇
    for k in range(4):
        ind = where(code == k)[0]
        figure()
        gray()
        for i in range(minimum(len(ind), 40)):
            subplot(4, 10, i + 1)
            imshow(immatrix[ind[i]].reshape((25, 25)))
            axis('off')
    show()


def main3():
    """
     在主成分上可视化图像
    :return:
    """
    imlist = imtools.get_imlist('../resource/picture/selectedfontimages/a_selected_thumbs')
    imnbr = len(imlist)

    # Load images, run PCA.
    immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')
    V, S, immean = pca.pca(immatrix)

    projected = array([dot(V[[0, 1]], immatrix[i] - immean) for i in range(imnbr)])

    # height and width
    h, w = 1200, 1200

    # create a new image with a white background
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # draw axis
    draw.line((0, h / 2, w, h / 2), fill=(255, 0, 0))
    draw.line((w / 2, 0, w / 2, h), fill=(255, 0, 0))

    # scale coordinates to fit
    scale = abs(projected).max(0)
    scaled = floor(array([(p / scale) * (w / 2 - 20, h / 2 - 20) + (w / 2, h / 2)
                          for p in projected])).astype(int)

    # paste thumbnail of each image
    for i in range(imnbr):
        nodeim = Image.open(imlist[i])
        nodeim.thumbnail((25, 25))
        ns = nodeim.size
        box = (scaled[i][0] - ns[0] // 2, scaled[i][1] - ns[1] // 2,
               scaled[i][0] + ns[0] // 2 + 1, scaled[i][1] + ns[1] // 2 + 1)
        img.paste(nodeim, box)

    tree = hcluster.hcluster(projected)
    hcluster.draw_dendrogram(tree, imlist, filename='fonts.png')

    figure()
    imshow(img)
    axis('off')
    img.save('pca_font.png')
    show()


def clusterpixels(infile, k, steps):
    # 图像被划分成steps x steps的区域
    im = array(Image.open(infile))
    dx = im.shape[0] // steps
    dy = im.shape[1] // steps

    # 计算每个区域的颜色特征
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = array(features, 'f')  # 变为数组

    # 聚类
    centroids, variance = kmeans(features, k)
    code, distance = vq(features, centroids)

    codeim = code.reshape(steps, steps)
    # codeim = imresize(codeim, im.shape[:2], 'nearest')
    # 我他妈的真是无语了，scipy.misc.imresize模块早就被移除了
    # cv2.resize可以实现同样的效果
    codeim = cv2.resize(src=codeim, dsize=im.shape[:2], interpolation=cv2.INTER_NEAREST)
    return codeim


def main4():
    k = 3
    infile_empire = '../resource/picture/empire.jpg'
    infile_boy_on_hill = '../resource/picture/boy_on_hill.jpg'
    im_empire = array(Image.open('../resource/picture/empire.jpg'))
    im_boy_on_hill = array(Image.open('../resource/picture/boy_on_hill.jpg'))

    # 显示原图empire.jpg
    figure()
    # 231代表2行3列第1个
    subplot(231)
    title('empire.jpg')
    axis('off')
    imshow(im_empire)

    # 用50*50的块对empire.jpg的像素进行聚类
    codeim = clusterpixels(infile_empire, k, 50)
    subplot(232)
    title('k=3,steps=50')
    axis('off')
    imshow(codeim)

    # 用100*100的块对empire.jpg的像素进行聚类
    codeim = clusterpixels(infile_empire, k, 100)
    subplot(233)
    title('k=3,steps=100')
    axis('off')
    imshow(codeim)

    # 显示原图empire.jpg
    subplot(234)
    title('boy_on_hill.jpg')
    axis('off')
    imshow(im_boy_on_hill)

    # 用50*50的块对empire.jpg的像素进行聚类
    codeim = clusterpixels(infile_boy_on_hill, k, 50)
    subplot(235)
    title('k=3,steps=50')
    axis('off')
    imshow(codeim)

    # 用100*100的块对empire.jpg的像素进行聚类
    codeim = clusterpixels(infile_boy_on_hill, k, 100)
    subplot(236)
    title('k=3，steps=100')
    axis('off')
    imshow(codeim)
    show()


if __name__ == '__main__':
    # main1()
    # main2()
    # main3()
    main4()
