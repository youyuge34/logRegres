# encoding: utf-8
"""
@author: yousheng
@contact: 1197993367@qq.com
@site: http://youyuge.cn

@version: 1.0
@license: Apache Licence
@file: test.py
@time: 17/9/5 上午10:14

"""
import logRegres as lr

def run():
    dataMat,labelMat = lr.loadDataSet()
    weights = lr.stocGradAscent1(dataMat,labelMat)
    print weights
    lr.plotBestFit(weights)


def runColic():
    """
    疝气病病马案例测试
    :return:
    """
    lr.multiTest()


if __name__ == '__main__':
    # run()
    runColic()