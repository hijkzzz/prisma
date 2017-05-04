import os
import tensorflow as tf
from scipy import misc
import vgg
import transform

tf.app.flags.DEFINE_string("MODEL_PATH", "models/fast-style-model.ckpt-done", "Pre-trained models")
tf.app.flags.DEFINE_string("CONTENT_IMAGE", "raw-images/content-image.png", "Path to content image")
tf.app.flags.DEFINE_string("OUTPUT_FOLDER", "output-images/", "Path to output image")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1, "Number of concurrent images to train on")

FLAGS = tf.app.flags.FLAGS


def generate():
    if not FLAGS.CONTENT_IMAGE:
        tf.logging.info("train a fast nerual style need to set the Content images path")
        return

    if not os.path.exists(FLAGS.OUTPUT_FOLDER):
        os.mkdir(FLAGS.OUTPUT_FOLDER)

    # 获取图片信息
    height = 0
    width = 0
    with open(FLAGS.CONTENT_IMAGE, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.CONTENT_IMAGE.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default(), tf.Session() as sess:
        # 读取图片文件
        path = FLAGS.CONTENT_IMAGE
        png = path.lower().endswith('png')
        img_bytes = tf.read_file(path)

        # 图片解码
        content_image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
        content_image = tf.image.convert_image_dtype(content_image, tf.float32) * 255.0
        content_image = tf.expand_dims(content_image, 0)

        generated_images = transform.net(content_image - vgg.MEAN_PIXEL, training=False)
        output_format = tf.saturate_cast(generated_images, tf.uint8)

        # 开始转换
        saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        model_path = os.path.abspath(FLAGS.MODEL_PATH)
        tf.logging.info('Usage model {}'.format(model_path))
        saver.restore(sess, model_path)

        filename = os.path.basename(FLAGS.CONTENT_IMAGE)
        (shotname, extension) = os.path.splitext(filename)
        filename = shotname + '-' + os.path.basename(FLAGS.MODEL_PATH) + extension

        tf.logging.info("image {}".format(filename))
        images_t = sess.run(output_format)

        assert len(images_t) == 1
        misc.imsave(os.path.join(FLAGS.OUTPUT_FOLDER, filename), images_t[0])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    generate()