import cherrypy, os, urllib, pickle
import programming_computer_vision_with_python.ch07.imagesearch as imagesearch
from PCV.tools.imtools import get_imlist
from PCV.localdescriptors import sift
import programming_computer_vision_with_python.ch07.Vocabulary as Vocabulary
from pylab import *
from numpy import *
import sqlite3 as sqlite
from PIL import Image


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


class SearchDemo(object):

    def __init__(self):
        # 载入图像列表
        with open('../resource/picture/convert_images_format_test/imlist.txt') as f:
            self.imlist = f.readlines()
        f.close()
        self.nbr_images = len(self.imlist)
        self.ndx = list(range(self.nbr_images))

        # 载入词汇
        with open('../resource/picture/first1000/vocabulary.pkl', 'rb') as f:
            self.voc = pickle.load(f)
        f.close()

        # 设置可以显示多少辐图像
        self.maxres = 15

        # html的头部和尾部
        self.header = """
            <!doctype html>
            <head>
            <title>Image search</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """

    def index(self, query=None):
        self.src = imagesearch.Searcher('web.db', self.voc)

        html = self.header
        html += """
            <br />
            Click an image to search. <a href='?query='> Random selection </a> of images.
            <br /><br />
            """
        if query:
            # 查询数据库，并获取前面的图像
            res = self.src.query(query)[:self.maxres]
            for dist, ndx in res:
                imname = self.src.get_filename(ndx)
                html += "<a href='?query=" + imname + "'>"
                html += "<img src='" + imname + "' width='200' />"
                html += "</a>"
        else:
            # 如果没有查询图像则随机显示一些图像
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query=" + imname + ">"
                html += "<img src=" + imname + " width='200' />"
                html += "</a>"

        html += self.footer
        return html

    index.exposed = True


def main1():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/convert_images_format_test/')
    nbr_images = len(imlist)
    # 获取特征列表
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]
    # 提取文件夹下图像的sift特征
    for i in range(nbr_images):
        process_image(imlist[i], featlist[i])


def main2():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/convert_images_format_test/')
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
    with open('../resource/picture/convert_images_format_test/vocabulary.pkl', 'wb') as f:
        pickle.dump(voc, f)
    print('vocabulary is:', voc.name, voc.nbr_words)


def main3():
    # 获取图像列表
    imlist = get_imlist('../resource/picture/convert_images_format_test/')
    nbr_images = len(imlist)
    # 获取特征列表
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]
    # 载入词汇
    with open('../resource/picture/convert_images_format_test/vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)

    # 创建索引表
    indx = imagesearch.Indexer('web.db', voc)
    indx.create_tables()

    # 遍历整个图像库，将特征投影到词汇上并添加到索引中
    for i in range(nbr_images)[:1000]:
        locs, descr = sift.read_features_from_file(featlist[i])
        indx.add_to_index(imlist[i], descr)
    # 提交到数据库
    indx.db_commit()

    con = sqlite.connect('web.db')
    print(con.execute('select count(filename) from imlist').fetchone())
    print(con.execute('select * from imlist').fetchone())


if __name__ == '__main__':
    # main1()
    # main2()
    # main3()
    # TODO 未调通
    cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
