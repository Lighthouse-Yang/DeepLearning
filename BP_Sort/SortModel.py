"""
    Author : YangBo
    Time : 2018-10-18 11:50
    function:神经网络实现分类.
"""
import tensorflow as tf
import numpy as np
import DataProcessing


if __name__ == '__main__':
    # 下载数据.
    raw_data = DataProcessing.download_data()
    # 数据编码及处理.
    data = DataProcessing.onehot_data(raw_data)
    data = data.values.astype(np.float32)
    # 将数据顺序打乱并将数据以8:2分为训练数据与测试数据.
    np.random.shuffle(data)
    scale = int(0.8*len(data))
    train_data = data[:scale]
    test_data = data[scale:]

    INPUT = tf.placeholder(tf.float32, [None, 25], "input")
    x = INPUT[:, :21]
    Y = INPUT[:, 21:]

    l1 = tf.layers.dense(x, 150, tf.nn.relu, name="l1")
    l2 = tf.layers.dense(l1, 150, tf.nn.relu, name="l2")
    OUTPUT = tf.layers.dense(l2, 4, name="output")
    # softmax回归.
    prediction = tf.nn.softmax(OUTPUT, name="pred")

    # 损失函数.
    loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=OUTPUT)
    # 准确度.
    accurary = tf.metrics.accuracy(
        labels=tf.argmax(Y, axis=1),predictions=tf.argmax(OUTPUT, axis=1),
    )[1]

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = opt.minimize(loss)

    sess = tf.Session()
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    for i in range(30001):
        batch_index = np.random.randint(len(train_data), size=32)
        sess.run(train_op, {INPUT: train_data[batch_index]})

        if i % 20 == 0:
            pred_, accurary_, loss_ = sess.run([prediction, accurary, loss], {INPUT:test_data})
            print("第{}轮训练,准确率:{},损失函数:{}".format(i, accurary_, loss_))


