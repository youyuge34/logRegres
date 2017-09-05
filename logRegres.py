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
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()