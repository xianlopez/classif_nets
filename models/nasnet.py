import tensorflow as tf
from models.nasnet_slim.nasnet import build_nasnet_large, build_nasnet_mobile, nasnet_large_arg_scope

slim = tf.contrib.slim

def build(inputs, nclasses, is_training, large):
    with slim.arg_scope(nasnet_large_arg_scope()):
        if large:
            nasnet = build_nasnet_large(inputs, nclasses, is_training=is_training)
        else:
            nasnet = build_nasnet_mobile(inputs, nclasses, is_training=True)
    logits = nasnet[0]
    return logits, logits

