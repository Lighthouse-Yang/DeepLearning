"""
    Author : YangBo
    Time : 2018-10-20 15:22
    function:
"""
import tensorflow as tf
import numpy as np
import Sort_forward
import Sort_backward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, Sort_forward.INPUT_NODE])
        y = Sort_forward.forward(x, None)
        preValue = tf.argmax(y, 1)    # 得到概率最大的预测值.
        """
            实现滑动平均模型,参数MOVING_AVERAGE_DECAY用于控制模型更新的速度.
            训练过程中会对每一个变量维护一个影子变量,这个影子变量的初始值就是相应变量的初始值,
            每次变量更新时,影子变量就会随之更新.    
        """
        variable_averages = tf.train.ExponentialMovingAverage(Sort_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            # 通过checkpoint文件定位到最新保存的模型.
            ckpt = tf.train.get_checkpoint_state(Sort_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print('找不到断点续训文件')
                return -1


# # 预处理函数,包括esize,转变灰度图,二值化操作.
# def pre_pic(picName):
#     img = Image.open(picName)
#     reIm = img.resize((28, 28), Image.ANTIALIAS)
#     im_arr = np.array(reIm.convert('L'))
#     threshold = 50    # 设定合理的阈值.
#     for i in range(28):
#         for j in range(28):
#             im_arr[i][j] = 255 - im_arr[i][j]
#             if (im_arr[i][j] < threshold):
#                 im_arr[i][j] = 0
#             else:im_arr[i][j] = 255
#
#     nm_arr = im_arr.reshape([1, 784])
#     nm_arr = nm_arr.astype(np.float32)
#     img_ready = np.multiply(nm_arr, 1.0/255.0)
#
#     return img_ready


# def application():
#     testNum = input('请输入测试图片数量：')
#     for i in range(int(testNum)):
#         testPic = input('测试图片路径：')
#         testPicArr = pre_pic(testPic)
#         preValue = restore_model(testPicArr)
#         print('预测结果为：{}'.format(preValue))


def main():
    application()


if __name__ == '__main__':
    main()