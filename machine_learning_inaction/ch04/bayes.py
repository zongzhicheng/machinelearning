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


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # p0Num = zeros(numWords)
    p0Num = ones(numWords)
    # p1Num = zeros(numWords)
    p1Num = ones(numWords)
    # p0Denom = 0.0
    p0Denom = 2.0
    # p1Denom = 0.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法
    # p1Vect = p1Num / p1Denom
    p1Vect = log(p1Num / p1Denom)
    # p0Vect = p0Num / p0Denom
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 元素相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocalList = createVocabList(listOPosts)
    print(myVocalList)
    print(setOfWords2Vec(myVocalList, listOPosts[0]))
    print(setOfWords2Vec(myVocalList, listOPosts[3]))

    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocalList, postinDoc))
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(pAb)
    # print(p0V)
    # print(p1V)

    testingNB()
