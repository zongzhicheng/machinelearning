from PIL import Image
from pylab import *
from numpy import *
from PCV.tools import imtools, pca
from scipy.cluster.vq import *


def main1():
    imlist = imtools.get_imlist('../resource/picture/selectedfontimages/a_selected_thumbs')
    imnbr = len(imlist)

    # Load images, run PCA.
    immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')
    V, S, immean = pca.pca(immatrix)

    # Project on 2 PCs.
    # projected = array([dot(V[[0, 1]], immatrix[i] - immean) for i in range(imnbr)])
    projected = array([dot(V[[1, 2]], immatrix[i] - immean) for i in range(imnbr)])

    n = len(projected)
    # 计算距离矩阵
    S = array([[sqrt(sum((projected[i] - projected[j]) ** 2))
                for i in range(n)] for j in range(n)], 'f')

    # 创建拉普拉斯矩阵
    rowsum = sum(S, axis=0)
    D = diag(1 / sqrt(rowsum))
    I = identity(n)
    L = I - dot(D, dot(S, D))

    # 计算矩阵L的特征向量
    U, sigma, V = linalg.svd(L)

    k = 5
    # 从矩阵L的前k个特征向量中创建特征向量
    # 叠加特征向量作为数组的列
    features = array(V[:k]).T

    # k-means聚类
    features = whiten(features)
    centroids, distortion = kmeans(features, k)
    code, distance = vq(features, centroids)

    # 绘制聚类簇
    for c in range(k):
        ind = where(code == c)[0]
        figure()
        gray()
        for i in range(minimum(len(ind), 39)):
            im = Image.open(imlist[ind[i]])
            subplot(4, 10, i + 1)
            imshow(array(im))
            axis('equal')
            axis('off')
    show()


if __name__ == '__main__':
    main1()
