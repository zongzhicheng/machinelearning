import sqlite3 as sqlite
import pickle


class Indexer(object):

    def __init__(self, db, voc):
        """初始化数据库的名称及词汇对象"""
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def get_id(self, imname):
        """
        获取图像id，如果不存在，就进行添加
        :param imname:
        :return:
        """
        cur = self.con.execute(
            "select rowid from imlist where filename='%s'" % imname)
        res = cur.fetchone()
        if res == None:
            cur = self.con.execute(
                "insert into imlist(filename) values ('%s')" % imname)
            return cur.lastrowid
        else:
            return res[0]

    def is_indexed(self, imname):
        """
        如果图像名字（imname）被索引到，就返回True
        :param imname:
        :return:
        """
        im = self.con.execute("select rowid from imlist where filename='%s'" % imname).fetchone()
        return im != None

    def add_to_index(self, imname, descr):
        """
        获取一幅带有特征描述子的图像，投影到词汇上并添加到数据库
        :param imname:
        :param descr:
        :return:
        """
        if self.is_indexed(imname):
            return
        print('indexing', imname)

        # 获取图像id
        imid = self.get_id(imname)

        # 获取单词
        imwords = self.voc.project(descr)
        nbr_words = imwords.shape[0]

        # 将每个单词与图像链接起来
        for i in range(nbr_words):
            word = imwords[i]
            # wordid就是单词本身的数字
            self.con.execute("insert into imwords(imid,wordid,vocname) values(?, ?, ?)",
                             (imid, word, self.voc.name))

            # 存储图像的单词直方图
            # 用pickle模块将Numpy数组编码成字符串
            self.con.execute("insert into imhistograms(imid,histogram,vocname) values (?,?,?)",
                             (imid, pickle.dumps(imwords), self.voc.name))

    def create_tables(self):
        """
        创建数据库表单
        :return:
        """
        self.con.execute('create table imlist(filename)')
        self.con.execute('create table imwords(imid,wordid,vocname)')
        self.con.execute('create table imhistograms(imid,histogram,vocname)')
        self.con.execute('create index im_idx on imlist(filename)')
        self.con.execute('create index wordid_idx on imwords(wordid)')
        self.con.execute('create index imid_idx on imwords(imid)')
        self.con.execute('create index imidhist_idx on imhistograms(imid)')
        self.db_commit()
