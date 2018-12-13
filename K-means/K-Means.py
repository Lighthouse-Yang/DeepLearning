"""
    Author : YangBo
    Time : 2018-12-05 21:11
    function:K-均值聚类法，进行数据分类.
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
"""
    数据集导入.
"""


def loadDataSet(filepath):
    datamat = []
    fr = open(filepath)
    for line in fr.readlines():
        curline = line.strip().split(',')   # 以','分割字符串且首尾删除空格(默认空格).
        # python3中map()返回的是迭代器,所以需要加list().
        # map():根据提供的参数对指定序列做映射.如将curLine序列映射为float类型.
        fltline = list(map(float, curline))
        datamat.append(fltline)              # 将处理后的原始数据添加到列表datamat中.
    return datamat


def distEclud(vecA, vecB):
    return sqrt(np.sum(power(np.array(vecA) - np.array(vecB), 2)))


def randCent(dataset, k):                     # K:表示所分簇的个数.
    n = shape(dataset)[1]                     # shape()[] 得到矩阵的纬度.
    centroids = mat(zeros((k, n)))              # 构建一个K行N列的零矩阵.
    for j in range(n):
        ar = np.array(dataset)
        minJ = min(ar[:, j])
        rangeJ = float(max(ar[:, j]) - minJ)   # 最大值-最小值=取值范围.
        # 生成(0,1.0)的随机数,由minJ和rangJ控制,生成K个在边界内的随机质心点.
        centroids[:, j] = (minJ + rangeJ * random.rand(k, 1))
    return centroids


def KMeans(dataset, k):
    m = shape(dataset)[0]
    # clusterData:第一列:记录簇索引值.第二列:存储误差.
    clusterData = mat(zeros((m, 2)))
    # 调用randCent函数获取随机生成的质心.
    centroids = randCent(dataset, k)
    count = 1
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1  # 初始化最小值.
            for j in range(k):
                # 调用distEclud函数计算距离.
                distJI = distEclud(np.array(centroids)[j, :], np.array(dataset)[i, :])
                if distJI < minDist:
                    # 更新簇信息.
                    minDist = distJI; minIndex = j
            # 当簇索引值不再变化时,即表示聚类稳定.
            if clusterData[i, 0] != minIndex:
                clusterChanged = True
            clusterData[i, :] = minIndex, minDist**2
        print("第{}次迭代质心:\n{}".format(count, centroids))
        for cent in range(k):
            ptsInClust = np.array(dataset)[nonzero(clusterData[:, 0].A==cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
        plotKmens(dataset, centroids)
        count = count + 1
    return centroids, clusterData


def plotKmens(dataset, centroids):
    # 绘制聚类结果.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.array(dataset)[:, 0], np.array(dataset)[:, 1], c='blue')
    ax.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='red', marker='+', s=70)
    plt.show()


if __name__ == '__main__':
    filepath = 'D:/pycharm/code/python_learning_test/K-means/data.txt'
    datamat = loadDataSet(filepath)
    KMeans(datamat, 3)