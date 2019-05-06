import tensorflow as tf
import time


def speed_test_1():
    n_steps = 10
    size = 10000

    x = tf.Variable(initial_value=tf.random.normal(shape=(size, size)))
    y = tf.Variable(initial_value=tf.random.normal(shape=(size, size)))
    ass_op = x.assign(tf.matmul(x, y))

    with tf.Session() as sess:
        sess.run(x.initializer)
        sess.run(y.initializer)
        for i in range(n_steps):
            ini = time.time()
            sess.run(ass_op)
            fin = time.time()
            print('Step ' + str(i) + ' done in ' + str(fin - ini) + ' s.')


if __name__ == '__main__':
    speed_test_1()
