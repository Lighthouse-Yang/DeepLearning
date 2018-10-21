"""
    Author : YangBo
    Time : 2018-10-21 19:00
    function:reduce_mean函数.
"""
import tensorflow as tf
import numpy as np
a = [1., 2., 3., 4.]
# 2行4列矩阵.
b = np.array([[1., 3., 4., 4.], [2., 3., 4., 5.]])

sess = tf.Session()
reduce_mean = sess.run(tf.reduce_mean(a))
# 列维度上求平均值.
reduce_mean0 = sess.run(tf.reduce_mean(b, 0))
# 行维度上求平均值.
reduce_mean1 = sess.run(tf.reduce_mean(b, 1))
# 求最大值.
reduce_max = sess.run(tf.reduce_max(a))
reduce_min = sess.run(tf.reduce_min(a))
reduce_sum = sess.run(tf.reduce_sum(a))
print(reduce_mean)
print(reduce_mean0)
print(reduce_mean1)
print(reduce_max)
print(reduce_min)
print(reduce_sum)