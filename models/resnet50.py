# ======================================================================================================================
import tensorflow as tf
import tensorflow.contrib.slim as slim


# ----------------------------------------------------------------------------------------------------------------------
def build(inputs, nclasses, is_training):

    # We use TensorFlow's SLIM high level API to obtain ResNet50 architecture:
    resnet_v1 = tf.contrib.slim.nets.resnet_v1
    with slim.arg_scope(tf.contrib.slim.python.slim.nets.resnet_utils.resnet_arg_scope()): # This arg scope is mandatory. Otherwise we checkpoint file will fail at loading
        logits, _ = resnet_v1.resnet_v1_50(inputs, num_classes=nclasses, is_training=is_training, scope='resnet_v1_50')
        logits = tf.squeeze(logits, axis=[1, 2])

    return logits, logits