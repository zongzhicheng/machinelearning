# 《机器学习实战》
# 朴素贝叶斯
# --------------------------------
# ---------- 2021.1.15 ----------
# --------------------------------
from numpy import *
import re


# 词表到向量的转换函数
def loadDataSet():
    # 切分的词条
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
# ['my', 'flea', 'posting', 'dog', 'has', 'dalmation', 'not',
# 'problems', 'steak', 'stupid', 'I', 'help', 'buying', 'licks',
# 'him', 'so', 'garbage', 'how', 'to', 'park', 'quit', 'maybe',
# 'take', 'worthless','cute', 'is', 'food', 'ate', 'love', 'please', 'mr', 'stop']
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)


# 检查词汇表中的单词在输入文档中是否出现
# 朴素贝叶斯词集模型
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的元素
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回文档向量


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    # 1是侮辱类 0是非侮辱类
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率

    # 词条出现数初始化
    # p0Num = zeros(numWords)
    p0Num = ones(numWords)
    # p1Num = zeros(numWords)
    p1Num = ones(numWords)

    # 分母初始化
    # p0Denom = 0.0
    p0Denom = 2.0
    # p1Denom = 0.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率，即P(w0|1)、P(w1|1)、P(w2|1)...
            p1Num += trainMatrix[i]  # 将统计所有侮辱类每个单词出现个数
            p1Denom += sum(trainMatrix[i])  # 所有侮辱类文档中所有单词出现个数
        else:  # 统计属于非侮辱类的条件概率，即P(w0|0)、P(w1|0)、P(w2|0)...
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法
    # 这里使用log是因为用python乘许多很小的数，最终四舍五入等于0
    # p1Vect = p1Num / p1Denom
    p1Vect = log(p1Num / p1Denom)  # 所有侮辱类文档中每个单词出现概率即P(wi|1)
    # p0Vect = p0Num / p0Denom
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯公式计算
# 这里假设所有词都是互相独立的，P(ci|wi)=P(wi|ci)P(ci)
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 元素相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 朴素贝叶斯分类函数
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


# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 将字符串转换为字符列表
def textParse(bigString):
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    listOfTokens = re.split(r'\\W*', bigString)
    # 除了单个字母，例如大写的I，其它单词变成小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 垃圾邮件测试函数
def spamTest():
    docList = []
    classList = []
    fullText = []
    # 遍历25个文件
    for i in range(1, 26):
        # 读取每个垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        # append和extend区别
        """
        >>> li = ['a', 'b', 'c']  
        >>> li.extend(['d', 'e', 'f'])   
        >>> li  
        ['a', 'b', 'c', 'd', 'e', 'f']  
        """
        """
        >>> li = ['a', 'b', 'c']  
        >>> li.append(['d', 'e', 'f'])   
        >>> li  
        ['a', 'b', 'c', ['d', 'e', 'f']]  
        """
        docList.append(wordList)
        fullText.extend(wordList)
        # 标记垃圾邮件，1表示垃圾文件
        classList.append(1)
        # 读取每个非垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        # 标记非垃圾邮件，0表示非垃圾文件
        classList.append(0)
    # 创建词汇表，不重复
    vocabList = createVocabList(docList)
    # 创建存储训练集的索引值的列表
    trainingSet = list(range(50))
    # 创建测试集的索引值的列表
    testSet = []
    # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
    for i in range(10):
        # 随机选取索索引值
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        # 在训练集列表中删除添加到测试集的索引值
        del (trainingSet[randIndex])
    # 创建训练集矩阵
    trainMat = []
    # 创建训练集类别标签系向量
    trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 将生成的词集模型添加到训练矩阵中
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        # 将类别添加到训练集类别标签系向量中
        trainClasses.append(classList[docIndex])
    # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 错误分类计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:
        # 测试集的词集模型
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 如果分类错误
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集", docList[docIndex])
    print('错误率: ', float(errorCount) / len(testSet))


if __name__ == '__main__':
    """
    listOPosts, listClasses = loadDataSet()
    myVocalList = createVocabList(listOPosts)
    print(myVocalList)
    for postinDoc in listOPosts:
        print(postinDoc)
        print(setOfWords2Vec(myVocalList, postinDoc))

    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocalList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print(pAb)  # 文档属于侮辱类的概率P(c1)
    print(p0V)  # 所有侮辱类文档中每个单词出现的概率，即P(wi|c1)
    print(p1V)  # 所有非侮辱类文档中每个单词出现的概率P，即(wi|c0)
    """

    testingNB()

    spamTest()
