import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_iterator():
    def _parser_func(lr_image, hr_image):
        lr_image = tf.image.decode_image(tf.read_file(lr_image))
        lr_image = tf.squeeze(lr_image)
        lr_image = tf.extract_image_patches(tf.expand_dims(lr_image, 0), [1, 112, 112, 1], [1, 112, 112, 1],
                                            [1, 1, 1, 1], "VALID")
        lr_image = tf.reshape(lr_image, [-1, 112, 112, 3])
        hr_image = lr_image
        lr_image = tf.image.resize_images(lr_image, (28, 28), method=1, align_corners=True)
        lr_image = tf.cast(tf.divide(lr_image, 255), dtype=tf.float32)
        hr_image = tf.cast(tf.divide(hr_image, 255), dtype=tf.float32)
        return lr_image, hr_image

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.list_files("../DIV2K_train_HR/*.png")
        dataset = dataset.shuffle(1000)
        dataset = dataset.zip((dataset, dataset))
        dataset = dataset.map(_parser_func)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        dataset = dataset.batch(16)
        dataset = dataset.repeat()

        iterator = dataset.make_initializable_iterator()
        return iterator


def get_eval_iterator():
    def _parser_func(lr_image, hr_image):
        lr_image = tf.image.decode_image(tf.read_file(lr_image))
        lr_image = tf.squeeze(lr_image)
        lr_image = tf.extract_image_patches(tf.expand_dims(lr_image, 0), [1, 112, 112, 1], [1, 112, 112, 1],
                                            [1, 1, 1, 1], "VALID")
        lr_image = tf.reshape(lr_image, [-1, 112, 112, 3])
        hr_image = lr_image
        lr_image = tf.image.resize_images(lr_image, (28, 28), method=1, align_corners=True)
        lr_image = tf.cast(tf.divide(lr_image, 255), dtype=tf.float32)
        hr_image = tf.cast(tf.divide(hr_image, 255), dtype=tf.float32)
        return lr_image, hr_image

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.list_files("../DIV2K_valid_HR/*.png")
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.zip((dataset, dataset))
        dataset = dataset.map(_parser_func)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        dataset = dataset.batch(16)

        iterator = dataset.make_initializable_iterator()
        return iterator


def get_inference_iterator(inference_img):
    def _parser_func(image):
        image = tf.squeeze(image)
        image = tf.extract_image_patches(tf.expand_dims(image, 0), [1, 28, 28, 1], [1, 28, 28, 1], [1, 1, 1, 1], "VALID")
        #image = tf.cast(tf.divide(image, 255), dtype=tf.float32)
        image = tf.reshape(image, [-1, 28, 28, 3])
        return image

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensors(mpimg.imread(inference_img)[:, :, 0:3])
        dataset = dataset.map(_parser_func)
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices((x)))
        dataset = dataset.batch(1)

        iterator = dataset.make_one_shot_iterator()
        return iterator


if __name__ == "__main__":
    iterator = get_eval_iterator()
    next_element = iterator.get_next()
    i = 0
    with tf.Session() as sess:
        image_batches = 0
        while True:
            try:
                sess.run(iterator.initializer)
                lr_image, hr_image = sess.run(next_element)
                batch = np.shape(lr_image)[0]
                i += batch
                plt.imshow(hr_image[0])
                plt.show()
                print(i)
            except tf.errors.OutOfRangeError:
                print(i)
                break
