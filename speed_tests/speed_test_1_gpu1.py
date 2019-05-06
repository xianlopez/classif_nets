from speed_tests import speed_test_1
import tensorflow as tf

with tf.device('/gpu:1'):
    speed_test_1.speed_test_1()
