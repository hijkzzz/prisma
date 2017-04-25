# coding=utf-8

import tensorflow as tf
import vgg
import reader


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / \
        tf.to_float(width * height * num_filters)

    return grams


def get_style_features(style_paths, style_layers, image_size, style_scale, vgg_path):
    with tf.Graph().as_default(), tf.Session() as sess:
        size = int(round(image_size * style_scale))
        images = tf.stack(
            [reader.get_image(path, size) for path in style_paths])
        net, _ = vgg.net(vgg_path, images)
        features = []
        for layer in style_layers:
            features.append(gram(net[layer]))

        return sess.run(features)


def style_loss(net, style_features_t, style_layers):
    style_loss = 0
    for style_gram, layer in zip(style_features_t, style_layers):
        generated_images, _ = tf.split(net[layer], 2, 0)
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(
            gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss += layer_style_loss
    return style_loss


def content_loss(net, content_layers):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(
            net[layer], 2, 0)
        size = tf.size(generated_images)
        # remain the same as in the paper
        content_loss += tf.nn.l2_loss(generated_images -
                                      content_images) * 2 / tf.to_float(size)
    return content_loss


# 全变差正则化
def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack(
        [-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack(
        [-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + \
        tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
