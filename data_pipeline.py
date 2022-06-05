from math import radians

import tensorflow as tf
import tensorflow_addons as tfa


@tf.function
def process_labels(path, expression, valence, arousal):
    # labels = {"output_expression": expression,
    #           "output_valence": valence, "output_arousal": arousal}
    labels = expression
    return path, labels


@tf.function
def parse_img(path, *labels):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, *labels


def all_augment_img(image, *labels):
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, 0.6, 1.0)
    image = tf.image.adjust_gamma(
        image, tf.random.uniform(shape=[], minval=0.6, maxval=1.4))
    image = tf.image.random_jpeg_quality(image, 20, 100)
    image = tfa.image.rotate(image, tf.random.uniform(
        shape=[], minval=radians(-20), maxval=radians(20)))
    image = tfa.image.translate(image, [tf.random.uniform(shape=[], minval=-30, maxval=30),
                                        tf.random.uniform(shape=[], minval=-30, maxval=10)])
    return image, *labels


def random_flip_left_right(image, *labels):
    image = tf.image.random_flip_left_right(image)
    return image, *labels


def random_translate(image, *labels):
    image = tfa.image.translate(image, [tf.random.uniform(shape=[], minval=-30, maxval=30),
                                        tf.random.uniform(shape=[], minval=-30, maxval=10)])
    return image, *labels


def random_rotate(image, *labels):
    image = tfa.image.rotate(image, tf.random.uniform(
        shape=[], minval=radians(-20), maxval=radians(20)))
    return image, *labels


def random_hue(image, *labels):
    image = tf.image.random_hue(image, 0.01)
    return image, *labels


def random_contrast(image, *labels):
    image = tf.image.random_contrast(image, 0.6, 1.0)
    return image, *labels


def random_gamma(image, *labels):
    image = tf.image.adjust_gamma(
        image, tf.random.uniform(shape=[], minval=0.6, maxval=1.4))
    return image, *labels


def random_jpeg_quality(image, *labels):
    image = tf.image.random_jpeg_quality(image, 20, 100)
    return image, *labels
