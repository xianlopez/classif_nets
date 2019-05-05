# ======================================================================================================================
import tensorflow as tf
import tensorflow.contrib.slim as slim


# ----------------------------------------------------------------------------------------------------------------------
def build(inputs, nclasses, is_training):

    inception = tf.contrib.slim.nets.inception
    with slim.arg_scope(tf.contrib.slim.python.slim.nets.resnet_utils.resnet_arg_scope()): # This arg scope is mandatory. Otherwise we checkpoint file will fail at loading
        logits, _ = inception.inception_v3(inputs, num_classes=nclasses, is_training=is_training)
        print('logis after inception_v3')
        print(logits)
        # logits = tf.squeeze(logits, axis=[1, 2])
        # print('logis after squeeze')
        # print(logits)

    return logits, logits