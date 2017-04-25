# coding=utf-8

import os
import tensorflow as tf
from scipy import misc
import vgg
import transform
import reader

tf.app.flags.DEFINE_string("MODEL_FILE", "models/fast-style-model-done.ckpt", "Pre-trained models")
tf.app.flags.DEFINE_string("CONTENT_IMAGE", None, "Path to content image")
tf.app.flags.DEFINE_string("OUTPUT_PATH", "output/", "Path to output image(s)")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1, "Number of concurrent images to train on")
tf.app.flags.DEFINE_string("DEVICE", "/cpu:0", "Device for training")

FLAGS = tf.app.flags.FLAGS


def generate():
    if not FLAGS.CONTENT_IMAGE:
        tf.logging.info("train a fast nerual style need to set the Content images path")
        return

    # 获取图片信息
    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default(), tf.device(FLAGS.DEVICE), tf.Session() as sess:
        content_image = reader.get_image(FLAGS.CONTENT_IMAGE, max(height, width))
        generated_images = transform.net(content_image)
        output_format = tf.saturate_cast(generated_images + vgg.MEAN_PIXEL, tf.uint8)

        # 开始转换
        saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver.restore(sess, FLAGS.MODLE_FILE)

        filename = os.path.basename(FLAGS.CONTENT_IMAGE)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                tf.logging.info("image {}".format(filename))
                images_t = sess.run(output_format)

                for raw_image in images_t:
                    misc.imsave(os.path.join(FLAGS.OUTPUT_PATH,
                                'output-' + filename), raw_image)
        except tf.errors.OutOfRangeError:
            tf.logging.info(' Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    generate()