# coding=utf-8

import os
import time
import tensorflow as tf
import vgg
import transform
import loss
import reader


tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 7.5e0,
                            "Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 1e2,
                            "Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 2e2,
                            "Weight for total variation loss")
tf.app.flags.DEFINE_integer("LEARNING_RATE", 1e-3,
                            "Learning rate for training")
tf.app.flags.DEFINE_integer("EPOCHS", 2,
                            "Num epochs")
tf.app.flags.DEFINE_string("STYLE_IMAGES", "style-image.png", "Styles to train")
tf.app.flags.DEFINE_float("STYLE_SCALE", 1.0,
                          "Scale styles. Higher extracts smaller features")
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1,
                            "Number of concurrent images to train on")
tf.app.flags.DEFINE_string("MODEL_PATH", "models/",
                           "Path to read/write trained models")
tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
                           "Path to vgg model weights")
tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "train2014/",
                           "Path to training images")
tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu4_2",
                           "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("STYLE_LAYERS",
                           "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
                           "Which layers to extract style from")

FLAGS = tf.app.flags.FLAGS

# how to select GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def optimize():
    if not os.path.exists(FLAGS.MODEL_PATH):
        os.mkdir(FLAGS.MODEL_PATH)

    style_paths = FLAGS.STYLE_IMAGES.split(',')
    style_layers = FLAGS.STYLE_LAYERS.split(',')
    content_layers = FLAGS.CONTENT_LAYERS.split(',')

    # style gram matrix
    style_features_t = loss.get_style_features(style_paths, style_layers,
                                                FLAGS.IMAGE_SIZE, FLAGS.STYLE_SCALE, FLAGS.VGG_PATH)

    with tf.Graph().as_default(), tf.Session() as sess:
        # train_images
        images = reader.image(FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE,
                            FLAGS.TRAIN_IMAGES_PATH, FLAGS.EPOCHS)

        generated = transform.net(images / 255.0)
        net, _ = vgg.net(FLAGS.VGG_PATH, tf.concat([generated, images], 0))

        # 损失函数
        content_loss = loss.content_loss(net, content_layers)
        style_loss = loss.style_loss(net, style_features_t, style_layers) / len(style_paths)

        total_loss = FLAGS.STYLE_WEIGHT * style_loss + FLAGS.CONTENT_WEIGHT * content_loss + \
            FLAGS.TV_WEIGHT * loss.total_variation_loss(generated)

        # 准备训练
        global_step = tf.Variable(0, name="global_step", trainable=False)

        variable_to_train = []
        for variable in tf.trainable_variables():
            if not variable.name.startswith('vgg19'):
                variable_to_train.append(variable)

        train_op = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE).minimize(
            total_loss, global_step=global_step, var_list=variable_to_train)

        variables_to_restore = []
        for v in tf.global_variables():
            if not v.name.startswith('vgg19'):
                variables_to_restore.append(v)

        # 开始训练
        saver = tf.train.Saver(variables_to_restore,
                                write_version=tf.train.SaverDef.V2)
        sess.run([tf.global_variables_initializer(),
                    tf.local_variables_initializer()])

        # 加载检查点
        ckpt = tf.train.latest_checkpoint(FLAGS.MODEL_PATH)
        if ckpt:
            tf.logging.info('Restoring model from {}'.format(ckpt))
            saver.restore(sess, ckpt)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        try:
            while not coord.should_stop():
                _, loss_t, step = sess.run([train_op, total_loss, global_step])
                elapsed_time = time.time() - start_time
                start_time = time.time()

                if step % 10 == 0:
                    tf.logging.info(
                        'step: %d,  total loss %f, secs/step: %f' % (step, loss_t, elapsed_time))

                if step % 1000 == 0:
                    saver.save(sess, os.path.join(FLAGS.MODEL_PATH,
                                                    'fast-style-model.ckpt'), global_step=step)
                    tf.logging.info('Save model')

        except tf.errors.OutOfRangeError:
            saver.save(sess, os.path.join(
                FLAGS.MODEL_PATH, 'fast-style-model-done.ckpt'))
            tf.logging.info('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    optimize()
