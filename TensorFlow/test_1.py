"""
    Author : YangBo
    Time : 2018-09-17 14:09
    function:TensorFlow第一次练习.
"""
import tensorflow as tf


a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

result = a + b
print(result)

# with会话：进行结果计算.
with tf.Session() as sess:
    print(sess.run(result))
zero = tf.zeros([10, 10])
print(zero)
with tf.Session() as sess:
    print(sess.run(zero))