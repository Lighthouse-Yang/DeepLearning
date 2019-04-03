"""
    Author : YangBo
    Time : 2018-12-09 16:11
    function:K邻近算法实现UCI数据集分类.
"""
from numpy import *
import numpy as np
from collections import Counter


def datasetload(filepath):
    data = []
    labels = []
    fr = open(filepath)
    for line in fr.readlines():
        curline = line.strip().split(',')
        fltline = list(map(float, curline[:5]))
        data.append(fltline)
        labels.append(curline[5:])
    return data, labels


# 计算欧式距离衡量数据间的相似度.
def distEclud(vecA, vecB):
    return sqrt(np.sum(power(np.array(vecA) - np.array(vecB), 2)))


def KNN(K):
    exact_amount = 0
    for i in range(len(goal_data)):
        distance = {}   # distance字典: key(数据编号):value(欧式距离).
        K_labels = []   # 存储前K个邻近数据的类标签.
        for j in range(len(train_data)):
            distance.update({j+1: (distEclud(goal_data[i], train_data[j]))})
        # 对计算结果进行排序,找出最邻近的前K个值.
        distance = sorted(distance.items(), key=lambda x: x[1])
        for t in range(K):
            # 取出前K个邻近数据的编号及类标签构成K_labels.
            # print("前K个邻近数据排序:{}".format(distance[t][0]))
            K_labels.append((train_labels[(distance[t][0])-1])[0])
        # counts_label邻近数据中频率最大的类标签及频率.
        counts_label = Counter(K_labels).most_common(1)
        print("第{}条目标数据预测类别为:{}---正确类别为：{}".format(i+1, (counts_label[0])[0], goal_labels[i][0]))
        if (counts_label[0])[0] == ((goal_labels[i])[0]):
            exact_amount = exact_amount + 1
    # 准确率计算.
    accuracy = format(float(exact_amount) / float(len(goal_data)), '.2f')
    print("程序分类准确率：{}".format(accuracy))
    print("------------------分隔符------------------")


if __name__ == '__main__':
    train_path = 'D:/pycharm/code/python_learning_test/KNN/TRAIN_DATA.txt'
    goal_path = 'D:/pycharm/code/python_learning_test/KNN/GOAL_DATA.txt'
    train_data, train_labels = datasetload(train_path)
    goal_data, goal_labels = datasetload(goal_path)
    # 定义K邻近算法参数K.
    for K in range(1, 9):
        KNN(K)


