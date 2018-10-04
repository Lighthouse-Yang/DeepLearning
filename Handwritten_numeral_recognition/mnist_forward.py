"""
    Author : YangBo
    Time : 2018-09-28 14:18
    function:前向传播.
"""
import tensorflow as tf

# 784个节点,对应784个像素点.
INPUT_NODE = 784
"""
    输出节点10个----输出样式为：[0,0,1,0,0,0,0,0,0,0]
"""
OUTPUT_NODE = 10
# 隐藏层节点500个.
LAYER1_NODE = 500


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    """
        参数解释：
                w1:从输入层到隐含层的权重系数.输入层784个,隐含层500个节点,所以w1是一个784*500的矩阵[784,500].
    """
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    # 偏值b1.
    b1 = get_bias([LAYER1_NODE])
    """
        参数解释：
                y1:为隐含层的输出结果,其由x*w1+b1所得.
    """
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    """
            参数解释：
                    w2:从隐含层到输出层的权重系数.隐含层500个节点,输出层19个,所以w2是一个500*10的矩阵[500,10].
    """
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
