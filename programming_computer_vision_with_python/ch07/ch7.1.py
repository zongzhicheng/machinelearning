from PIL import Image
from pylab import *
from numpy import *
import pickle
from PCV.tools.imtools import get_imlist
from PCV.localdescriptors import sift
from scipy.cluster.vq import *
import os
import programming_computer_vision_with_python.ch07.imagesearch as imagesearch
import sqlite3 as sqlite


class Vocabulary(object):

    def __init__(self, name):
        self.name = name
        self.voc = []
        self.idf = []
        self.trainingdata = []
        self.nbr_words = 0

    def train(self, featurefiles, k=100, subsampling=10):
        """
        用含有k个单词的K-means列出在featurefiles中的特征文件训练出一个词汇。
        对训练数据下采样可以加快训练速度
        :param featurefiles:
        :param k:
        :param subsampling:
        :return:
        """

        nbr_images = len(featurefiles)
        # 从文件中读取特征
        descr = []
        descr.append(sift.read_features_from_file(featurefiles[0])[1])
        descriptors = descr[0]  # 将所有的特征并在一起，以便后面进行K-means聚类
        for i in arange(1, nbr_images):
            descr.append(sift.read_features_from_file(featurefiles[i])[1])
            descriptors = vstack((descriptors, descr[i]))

        # k-means: 最后一个参数决定运行次数
        self.voc, distortion = kmeans(descriptors[::subsampling, :], k, 1)
        self.nbr_words = self.voc.shape[0]

        # 遍历所有的训练图像，并投影到词汇上
        imwords = zeros((nbr_images, self.nbr_words))
        for i in range(nbr_images):
            imwords[i] = self.project(descr[i])

        nbr_occurences = sum((imwords > 0) * 1, axis=0)

        self.idf = log((1.0 * nbr_images) / (1.0 * nbr_occurences + 1))
        self.trainingdata = featurefiles

    def project(self, descriptors):
        """
        将描述子投影到词汇上，以创建单词直方图
        :param descriptors:
        :return:
        """

        # 图像单词直方图
        imhist = zeros((self.nbr_words))
        words, distance = vq(descriptors, self.voc)
        for w in words:
            imhist[w] += 1

        return imhist

    def get_words(self, descriptors):
        """
        将描述符转换为词语
        :param descriptors:
        :return:
        """
        return vq(descriptors, self.voc)[0]


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
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # 现在把sift.exe直接加在项目里 这样就可以直接使用了
    cmmd = str(r"sift.exe " + imagename + " --output=" + resultname + " " + params)

    # 此处巨坑 要用win32 而不是win64
    # cmmd = str(r"D:\vlfeat-0.9.20\bin\win32\sift.exe " + imagename + " --output=" + resultname + " " + params)
    os.system(cmmd)
    print('processd', imagename, 'to', resultname)


# 生成1000张sift
def main1():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/first1000/')
    nbr_images = len(imlist)
    # 获取特征列表
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]
    # 提取文件夹下图像的sift特征
    for i in range(nbr_images):
        process_image(imlist[i], featlist[i])


# 创建词汇
def main2():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/first1000/')
    nbr_images = len(imlist)
    # 获取特征列表
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]
    # 生成词汇
    voc = Vocabulary('ukbenchtest')
    startTime = time.time()
    print('train begin')
    voc.train(featlist, 1000, 10)
    endTime = time.time()
    print('train end')
    print('spend time:' + str(endTime - startTime))
    with open('../resource/picture/first1000/vocabulary.pkl', 'wb') as f:
        pickle.dump(voc, f)
    print('vocabulary is:', voc.name, voc.nbr_words)


# 遍历每个ukbench数据库中的样本图像，并将其加入我们的索引
def main3():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/first1000/')
    nbr_images = len(imlist)
    # 获取特征列表
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]

    # 载入词汇
    with open('../resource/picture/first1000/vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)

    # 创建索引表
    indx = imagesearch.Indexer('test.db', voc)
    indx.create_tables()

    # 遍历整个图像库，将特征投影到词汇上并添加到索引中
    for i in range(nbr_images)[:1000]:
        locs, descr = sift.read_features_from_file(featlist[i])
        indx.add_to_index(imlist[i], descr)

    # 提交到数据库
    indx.db_commit()

    con = sqlite.connect('test.db')
    print(con.execute('select count(filename) from imlist').fetchone())
    print(con.execute('select * from imlist').fetchone())


if __name__ == '__main__':
    # main1()
    # main2()
    main3()
