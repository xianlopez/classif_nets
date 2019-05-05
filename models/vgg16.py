import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets


def build(inputs, nclasses, is_training):

    vgg = tf.contrib.slim.nets.vgg
    logits, _ = vgg.vgg_16(inputs, num_classes=nclasses, is_training=is_training)

    return logits, logits