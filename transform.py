# coding=utf-8

import tensorflow as tf


# 定义图像转换网络
# 网络基本结构 [ 下采样 -- 残差网络 -- 上采样 ]
def net(image, training):
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
        deconv1 = tf.nn.relu(_resize_conv2d(res5, 128, 64, 3, 2, training))
    with tf.variable_scope('deconv2'):
        deconv2 = tf.nn.relu(_resize_conv2d(deconv1, 64, 32, 3, 2, training))
    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(_conv2d(deconv2, 32, 3, 9, 1))

    y = (deconv3 + 1) * 127.5

    return y


def _conv2d(x, input_filters, output_filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('conv'):

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(
            shape, stddev=0.1), name='weight')
        convolved = tf.nn.conv2d(
            x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')
        normalized = _instance_norm(convolved)

        return normalized


def _resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''
    with tf.variable_scope('conv_transpose') as scope:
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(
            x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return _conv2d(x_resized, input_filters, output_filters, kernel, strides)


# 论文 "Instance Normalization: The Missing Ingredient for Fast Stylization"
# 提出采用 Instance Normalization 替代原来的 Batch Normalization 可以改善转换效果
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
