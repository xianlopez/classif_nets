import tensorflow as tf
import time
import numpy as np


def random_add(x):
    if x[0, 0, 0, 0] < 0:
        to_add = np.random.normal(0, 1, size=x.shape)
    else:
        to_add = np.zeros_like(x)
        for i in range(3):
            to_add += np.random.normal(-1, 1, size=x.shape)
    x += to_add
    x[0, 2, 1, 0] = x[0, 1, 2, 0] * 5.6
    return x


def speed_test_3():

    n_steps = 10
    batch_size = 16
    input_height = 224
    input_width = 448

    x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_height, input_width, 3))
    net = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.py_func(random_add, [net], tf.float32)
    net.set_shape((batch_size, input_height/2, input_width/2, 32))
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.sigmoid)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.sigmoid)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.py_func(random_add, [net], tf.float32)
    net.set_shape((batch_size, input_height/4, input_width/4, 64))
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    net = tf.reshape(net, [batch_size, -1])
    net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=net, units=10)

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
    speed_test_3()
