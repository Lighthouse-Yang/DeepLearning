"""
    Author : YangBo
    Time : 2018-09-18 11:37
    function:
"""
# coding:utf-8
# 导入模块,生成模拟数据集.
import tensorflow as tf
# 导入科学计算模块.
import numpy as np
# 一次导入神经网络的数据集数量.
BATCH_SIZE = 8
seed = 23455

# 基于seed产生随机数.
rng = np.random.RandomState(seed)
# 随机数返回32行2列的矩阵,表示32组 体积与重量数据作为输入数据集.
X = rng.rand(32, 2)
"""
    从X这个32行2列的矩阵中取出一行,
    判断如果和小于1,给Y赋值1------Y=1
    判断如果和不小于1,给Y赋值0----Y=0
"""
# 作为输入数据集的标签(正确答案)
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print('X:\n', X)
print('Y:\n', Y)

# 1定义神经网络的输入,参数和输出,定义前向传播过程.
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数及反向传播方法.
loss = tf.reduce_mean(tf.square(y-y_))
# (0.001)内部参赛为学习率.
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 生成会话,训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前(未经训练)的参数取值.
    print('w1\n', sess.run(w1))
    print('w2\n', sess.run(w2))
    print('\n')

    # 训练模型.
    # 训练次数.
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        # 迭代500次输出一次损失函数.
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('经过{}次迭代后，损失函数为：{}'.format(i, total_loss))

    # 输出训练后的参数取值.
    print('\n')
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))


