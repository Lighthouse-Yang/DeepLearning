"""
    Author : YangBo
    Time : 2018-09-18 15:45
    function:一层神经网络实现酸奶的预测.
"""
# coding=utf-8
# 预测多或预测少的影响一样.
# 导入模板,生成数据集.
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
# 函数关系式：y=x1+x2已知.
X = rdm.rand(32, 2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

# 1定义神经网络的输入,参数和输出,定义前向传播过程.
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 2定义损失函数及反向传播方法.
# 定义损失函数为MSE,反向传播方法为梯度下降.
loss_mse = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 3生成会话,训练STEPS轮.
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = (i*BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print('经过{}次迭代,w1为:'.format(i))
            print(sess.run(w1), '\n')
    print('最终函数系数(W1)为：\n', sess.run(w1))
"""
    最终拟合结果：Y(销量)=0.98x1+1.02x2
"""