"""
    Author : YangBo
    Time : 2018-09-17 16:04
    function:神经网络前向传播练习.
"""
# coding:utf-8
# 两层简单神经网络(全连接)
import tensorflow as tf

# 定义输入和参数
# x输入层：1*2矩阵
x = tf.constant([[0.7, 0.5]])
"""
    x = tf.placeholder(tf.float32, shape=(1, 2))
    placeholder()函数解析：
    函数原型：tf.placeholder(dtype, shape=None, name=None）
    使用说明：该函数用于得到传递进来的真实的训练样本，同时也可以理解为形参，用于定义过程，在执行的时候再赋具体的值。
    （相当于首先定义一个容器，包含容量、size等信息，真正调用的时候再往容器里面注入东西）
    
    注意：不必指定初始值，可以在运行时，通过Session.run 函数的参数”feed_dict={x : value}”进行赋值
    print('y in this code is:\n', sess.run(y, feed_dict={x: }))
    feed_dict作为字典,x为key值,矩阵[[0.7, 0.5]]作为value值,就这样对x就行了赋值.
    
    参数说明： 
    dtype：数据类型。常用的是tf.float32,tf.float64等数值类型 
    shape：数据形状。默认是None，就是一维值，也可以是多维，比如[1,2,3] 
    name：名称
"""
# 隐含层第一层进行权值初始化.
"""
    1.tf.random_normal([2, 3]：w1为2*3的标准随机正态分布.
    2.tf.random_normal([3, 1]：w1为3*1的标准随机正态分布.
    3.stddev=1:标准差为1.
    4.seed=1:随机种子为1.
"""
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程.
"""
    matmul(x, w1):表示x矩阵与w1矩阵的乘法.
"""
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果.
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print('y in this code is:\n', sess.run(y))
