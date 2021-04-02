from PIL import Image
from pylab import *
from numpy import *
import pickle
from PCV.tools.imtools import get_imlist
from PCV.localdescriptors import sift
import os
import programming_computer_vision_with_python.ch07.imagesearch as imagesearch
import programming_computer_vision_with_python.ch07.Vocabulary as Vocabulary
import sqlite3 as sqlite


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
    voc = Vocabulary.Vocabulary('ukbenchtest')
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


def main4():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/first1000/')
    nbr_images = len(imlist)
    # 获取特征列表
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]
    # 载入词汇
    with open('../resource/picture/first1000/vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)
    src = imagesearch.Searcher('test.db', voc)
    locs, descr = sift.read_features_from_file(featlist[0])
    iw = voc.project(descr)

    print('ask using a histogram...')
    result_1 = src.candidates_from_histogram(iw)[:10]
    print(result_1)


def main5():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/first1000/')
    nbr_images = len(imlist)
    # 获取特征列表
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]
    # 载入词汇
    with open('../resource/picture/first1000/vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)
    src = imagesearch.Searcher('test.db', voc)
    print('try a query...')
    print(src.query(imlist[0])[:10])


def main6():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/first1000/')
    nbr_images = len(imlist)
    # 获取特征列表
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]
    # 载入词汇
    with open('../resource/picture/first1000/vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)
    src = imagesearch.Searcher('test.db', voc)
    result = imagesearch.compute_ukbench_score(src, imlist[:200])
    print(result)

    nbr_result = 20
    res = [w[1] for w in src.query(imlist[0])[:nbr_result]]
    imagesearch.plot_results(src, res)


if __name__ == '__main__':
    # main1()
    main2()
    # main3()
    # 就离谱，代码一样，结果不一样
    # TODO：暂未找到原因
    # main4()
    # main5()
    # main6()
