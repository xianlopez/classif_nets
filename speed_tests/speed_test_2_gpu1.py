from speed_tests import speed_test_2
import tensorflow as tf

with tf.device('/gpu:1'):
    speed_test_2.speed_test_2()
