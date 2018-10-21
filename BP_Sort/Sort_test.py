"""
    Author : YangBo
    Time : 2018-10-20 15:22
    function:
"""
# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import Sort_forward
import Sort_backward
# 利用导入的time包,对程序限定5s的循环时间.
TEST_INTERVAL_SECS = 5
"""
    测试函数
"""


def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, Sort_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, Sort_forward.OUTPUT_NODE])
        y = Sort_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(Sort_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                # 断点续训.
                ckpt = tf.train.get_checkpoint_state(Sort_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print('第{}轮训练后,测试精确度为：{}'.format(global_step, accuracy_score))
                else:
                    print('找不到断点续练文件')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets('./', one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()