import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets


def speed_test_6():

    n_steps = 10
    batch_size = 16
    input_height = 224
    input_width = 448

    x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_height, input_width, 3))

    resnet_v1 = tf.contrib.slim.nets.resnet_v1
    with slim.arg_scope(tf.contrib.slim.python.slim.nets.resnet_utils.resnet_arg_scope()): # This arg scope is mandatory. Otherwise we checkpoint file will fail at loading
        logits, _ = resnet_v1.resnet_v1_50(x, num_classes=10, is_training=True, scope='resnet_v1_50')
        logits = tf.squeeze(logits, axis=[1, 2])

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(n_steps):
            x_np = np.random.normal(size=(batch_size, input_height, input_width, 3))
            ini = time.time()
            sess.run(fetches=[logits], feed_dict={x: x_np})
            fin = time.time()
            print('Step ' + str(i) + ' done in ' + str(fin - ini) + ' s.')


if __name__ == '__main__':
    speed_test_6()
