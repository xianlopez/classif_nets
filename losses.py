import tensorflow as tf
import logging


def cross_entropy(labels, logits):
    if len(labels.shape) != 1:
        err_msg = 'Labels rank not equal to one (%i)' % len(labels.shape)
        logging.error(err_msg)
        raise ValueError(err_msg)
    tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.losses.get_total_loss()
