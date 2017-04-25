# coding=utf-8

import os
import tensorflow as tf
from scipy import misc
import vgg
import transform
import reader

tf.app.flags.DEFINE_string("MODEL_FILE", "models/fast-style-model.ckpt", "Pre-trained models")
tf.app.flags.DEFINE_string("CONTENT_IMAGES_PATH", None, "Path to content image(s)")
tf.app.flags.DEFINE_string("OUTPUT_PATH", "output/", "Path to output image(s)")
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1, "Number of concurrent images to train on")
tf.app.flags.DEFINE_string("DEVICE", "/cpu:0", "Device for training")

FLAGS = tf.app.flags.FLAGS


def generate():
    if not FLAGS.CONTENT_IMAGES_PATH:
        tf.logging.info("train a fast nerual style need to set the Content images path")
        return

    # 要转换的图片
    content_images, filenames = reader.image(
            FLAGS.BATCH_SIZE,
            FLAGS.IMAGE_SIZE,
            FLAGS.CONTENT_IMAGES_PATH,
            epochs=1,
            shuffle=False,
            crop=False)

    generated_images = transform.net(content_images)
    output_format = tf.saturate_cast(generated_images + vgg.MEAN_PIXEL, tf.uint8)

    # 开始转换
    with tf.Graph().as_default(), tf.device(FLAGS.DEVICE), tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver.restore(sess, FLAGS.MODLE_FILE)

        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                tf.logging.info("image {}".format(filenames[i]))
                images_t = sess.run(output_format)

                for raw_image in images_t:
                    i += 1
                    misc.imsave(os.path.join(FLAGS.OUTPUT_PATH,
                                'output-' + filenames[i]), raw_image)
        except tf.errors.OutOfRangeError:
            tf.logging.info(' Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    generate()