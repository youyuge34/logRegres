# encoding: utf-8
"""
@author: yousheng
@contact: 1197993367@qq.com
@site: http://youyuge.cn

@version: 1.0
@license: Apache Licence
@file: logRegres.py
@time: 17/9/5 上午8:58

"""
import codecs
import random
import numpy as np

from numpy import mat, shape, ones, exp


def loadDataSet():
    """
    从txt文件中读取数据
    :return:
    """
    dataMat = []
    labelMat = []
    fr = codecs.open('testSet.txt')
    for line in fr.readlines():
        dataList = line.strip().split()
        dataMat.append([1.0, float(dataList[0]), float(dataList[1])])
        labelMat.append(int(dataList[2]))
    return dataMat, labelMat


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def gradAscent(dataMatIn, labelMat):
    """
    梯度上升法算出最佳系数
    :param dataMat:
    :param labelMat:
    :return:
    """
    dataMat = mat(dataMatIn)
    labelMat = mat(labelMat).transpose()

    m, n = shape(dataMat)
    alpha = 0.001  # 移动步长
    times = 500
    weights = ones((n, 1))  # 初始化回归系数为1
    for i in range(times):
        hx = sigmoid(dataMat * weights)
        errors = labelMat - hx
        weights += alpha * dataMat.transpose() * errors
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArray = np.array(dataMat)
    m, n = shape(dataArray)
    xrecord1 = []
    yrecord1 = []
    xrecord2 = []
    yrecord2 = []
    for i in range(len(labelMat)):
        if int(labelMat[i]) == 1:
            xrecord1.append([dataArray[i, 1]])
            yrecord1.append([dataArray[i, 2]])
        else:
            xrecord2.append([dataArray[i, 1]])
            yrecord2.append([dataArray[i, 2]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xrecord1, yrecord1, s=30, c='red', marker='s')
    ax.scatter(xrecord2, yrecord2, s=30, c='green')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def stocGradAscent0(dataMat, classLabel):
    """
    随机梯度上升算法，一次仅仅用一个样本点更新回归系数
    :param dataMat:
    :param classLabel:
    :return:
    """
    dataMat = np.array(dataMat)
    m, n = shape(dataMat)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        hx = sigmoid(np.sum(dataMat[i] * weights))
        error = classLabel[i] - hx
        weights += alpha * error * dataMat[i]
    return weights


def stocGradAscent1(dataMat, classLabel, numIter=150):
    """
    随机梯度上升优化，调整每次迭代时的alpha，随机选取更新
    :param dataMat:
    :param classLabel:
    :param numIter:
    :return:
    """
    dataMat = np.array(dataMat)
    m, n = shape(dataMat)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha每次迭代时需要调整
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取更新
            hx = sigmoid(np.sum(dataMat[randIndex] * weights))
            error = float(classLabel[randIndex]) - hx
            weights += alpha * error * dataMat[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    """
    以数据和回归系数，计算对应的sigmoid函数值
    :param inX:
    :param weights:
    :return:
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    """
    对疝气病病马死亡率的训练集，测试集
    :return: 错误率
    """
    frTrain = codecs.open('horseColicTraining.txt')
    frTest = codecs.open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArray = []
        for i in range(21):
            lineArray.append(float(currLine[i]))
        trainingSet.append(lineArray)  # 将每行变array后加入训练集
        trainingLabels.append(currLine[21])
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)  # 获得最佳回归系数

    errors = 0.0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArray = []
        for i in range(21):
            lineArray.append(float(currLine[i]))
        #测试test集的分类结果
        if int(classifyVector(np.array(lineArray),trainWeights)) != \
            int(currLine[21]):
            errors += 1
    errorRate = float(errors)/numTestVec
    print 'the error rate is %f' % errorRate
    return errorRate

def multiTest():
    """
    求病马案例10次测试的平均值
    :return:
    """
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after %d iterations, the average error rate is %f' % (numTests,errorSum/float(numTests))
