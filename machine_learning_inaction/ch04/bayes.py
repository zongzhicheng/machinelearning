# 《机器学习实战》
# 朴素贝叶斯
# --------------------------------
# ---------- 2021.1.15 ----------
# --------------------------------
from numpy import *


# 词表到向量的转换函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 代表侮辱性文字、0代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)


# 检查词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的元素
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocalList = createVocabList(listOPosts)
    print(myVocalList)
    print(setOfWords2Vec(myVocalList, listOPosts[0]))
    print(setOfWords2Vec(myVocalList, listOPosts[0]))
