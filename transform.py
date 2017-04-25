# coding=utf-8

import tensorflow as tf


# 定义图像转换网络
# 网络基本结构 [ 下采样 -- 残差网络 -- 上采样 ]
def net(image):
    with tf.variable_scope('conv1'):
        conv1 = tf.nn.relu(_conv2d(image, 3, 32, 9, 1))
    with tf.variable_scope('conv2'):
        conv2 = tf.nn.relu(_conv2d(conv1, 32, 64, 3, 2))
    with tf.variable_scope('conv3'):
        conv3 = tf.nn.relu(_conv2d(conv2, 64, 128, 3, 2))
    with tf.variable_scope('res1'):
        res1 = _residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = _residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = _residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = _residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = _residual(res4, 128, 3, 1)
    with tf.variable_scope('deconv1'):
        deconv1 = tf.nn.relu(_conv2d_transpose(res5, 128, 64, 3, 2))
    with tf.variable_scope('deconv2'):
        deconv2 = tf.nn.relu(_conv2d_transpose(deconv1, 64, 32, 3, 2))
    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(_conv2d_transpose(deconv2, 32, 3, 9, 1))

    y = deconv3 * 127.5

    return y


def _conv2d(x, input_filters, output_filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('conv') as scope:

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(
            shape, stddev=0.1), name='weight')
        convolved = tf.nn.conv2d(
            x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')
        normalized = _instance_norm(convolved)

        return normalized


def _conv2d_transpose(x, input_filters, output_filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('conv_transpose') as scope:

        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(
            shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        convolved = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[
                                           1, strides, strides, 1], padding=padding, name='conv_transpose')

        normalized = _instance_norm(convolved)
        return normalized


# 原论文采用的是 Batch Normalization 方法
# 论文 "Instance Normalization: The Missing Ingredient for Fast Stylization"
# 提出采用 Instance Normalization 可以提升转换效果
def _instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def _residual(x, filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('residual') as scope:
        conv1 = _conv2d(x, filters, filters, kernel, strides, padding=padding)
        conv2 = _conv2d(tf.nn.relu(conv1), filters, filters,
                       kernel, strides, padding=padding)
        residual = x + conv2

        return residual
