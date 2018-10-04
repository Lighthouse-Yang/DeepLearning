"""
    Author : YangBo
    Time : 2018-09-28 11:23
    function:数据集的下载与解压.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('data/',one_hot=True)
mnist = input_data.read_data_sets('data/',one_hot=True)
training_image = mnist.train.images
training_label = mnist.train.labels
n = 20
curr_img = np.reshape(training_image[n , :] , (28 , 28))
plt.matshow(curr_img , cmap = plt.get_cmap('gray'))
plt.show()
