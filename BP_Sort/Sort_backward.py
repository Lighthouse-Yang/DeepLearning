"""
    Author : YangBo
    Time : 2018-10-20 13:18
    function:反向传播.
"""
import tensorflow as tf
import numpy as np
import os
import Sort_forward
import DataProcessing

# 喂入数据量.
BATCH_SIZE = 20
# 初始学习率.
LEARNING_BATE_BASE = 0.1
# 学习率衰减率.
LEARNING_BATE_DECAY = 0.99
# 正则化系数.
REGULARIZER = 0.0001
# 训练轮数.
STEPS = 50000
# 滑动平均衰减率.
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径.
MODEL_SAVE_PATH = "D:\pycharm\code\DeepLearning\BP_Sort\Sort_model"
MODEL_NAME = "Car_Sort_MODEL"


def backward(train_data):
    # placeholder为训练参数占位.
    x = tf.placeholder(tf.float32, [None, Sort_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, Sort_forward.OUTPUT_NODE])
    y = Sort_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_BATE_BASE,
        global_step,
        train_data.num_examples / BATCH_SIZE,
        LEARNING_BATE_DECAY,
        staircase=True
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训.
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = train_data.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('第{}轮训练后,损失函数为：{}'.format(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    # 下载数据.
    raw_data = DataProcessing.download_data()
    # 数据编码及处理.
    data = DataProcessing.onehot_data(raw_data)
    data = data.values.astype(np.float32)
    # 将数据顺序打乱并将数据以8:2分为训练数据与测试数据.
    np.random.shuffle(data)
    scale = int(0.8 * len(data))
    train_data = data[:scale]
    test_data = data[scale:]
    INPUT = tf.placeholder(tf.float32, [None, 25], "input")
    # 此处数据读取不明白该怎么处理.
    """
        mnist数据集是将训练数据与测试数据分开,而这个案例是一张csv的表格，不明白怎么处理.
    """
    backward(train_data)


if __name__ == '__main__':
    main()