"""
    Author : YangBo
    Time : 2018-10-17 10:27
    function:数据下载与预处理过程.
"""
import pandas as pd
import os
from urllib.request import urlretrieve

def download_data():
    """
        判断目录下是否存在数据集，不存在则通过urlretrieve实现自动下载数据集.
    """
    # UCI数据集地址：https://archive.ics.uci.edu/ml/datasets/car+evaluation
    if os.path.isfile("D:\pycharm\code\DeepLearning\BP_Sort\car.csv"):
        print('Car.csv already exists')
    else:
        save_path = "D:\pycharm\code\DeepLearning\BP_Sort\car.csv"
        download_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        urlretrieve(download_path, save_path)
        print("Car.csv download successfully")
    # 使用pandas对数据进行简单预处理.
    columns_name = ["buying","maint","doors","persons","lug_boot","safety","class"]
    return pd.read_csv("car.csv", names=columns_name)

def onehot_data(raw_data):
    """
        通过pandas将原始数据转化为onehot(独热)编码.
        原理：将原始数据onehot编码，形成由25个向量构成的向量组.
    """
    return pd.get_dummies(raw_data,prefix=raw_data.columns)

# if __name__ == '__main__':
#         raw_data = download_data()
#         data = onehot_data(raw_data)
#         print(data.head())