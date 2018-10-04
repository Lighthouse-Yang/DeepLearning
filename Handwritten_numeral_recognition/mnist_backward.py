"""
    Author : YangBo
    Time : 2018-09-28 15:04
    function:反向传播.
"""
import tensorflow as tf
import input_data
import os
import mnist_forward

# 喂入数据量.
BATCH_SIZE = 200
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
MODEL_SAVE_PATH = "D:\pycharm\code\python_learning_test\Handwritten_numeral_recognition\MNIST_MODEL"
MODEL_NAME = "MNIST_MODEL"


def backward(mnist):
    # placeholder为训练参数占位.
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ =tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem +tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_BATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
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
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('第{}轮训练后,损失函数为：{}'.format(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets('./', one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()