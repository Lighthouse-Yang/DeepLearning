"""
    Author : YangBo
    Time : 2018-09-19 9:51
    function:一层实现神经网络的预测.
    New function:因产销关系，预测的多少产生的损失分为：产大于销(成本损失),销大于产(利润损失).
"""
# coding=utf-8
# 产销大小的不同，产生的损失不同.
# 导入模板,生成数据集.
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
# 函数关系式：y=ax1+bx2已知.
X = rdm.rand(32, 2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

# 1定义神经网络的输入,参数和输出,定义前向传播过程.
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 2定义损失函数及反向传播方法.
"""
    # 定义自定义损失函数,通过比较产销关系就行损失函数计算.
    COST = 1        # 酸奶成本.
    PROFIT = 9      # 酸奶利润.
    tf.where(tf.greater(y, y_)判断y与y_的大小关系：
                    y>y_:COST * (y - y_)
                    y<y_:PROFIT * (y_ - y)))
"""
COST = 1
PROFIT = 9
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST * (y - y_), PROFIT * (y_ - y)))
# 反向传播方法为梯度下降.
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

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
    最终拟合结果：Y(销量)=1.02x1+1.04x2
"""