import speed_test_1
import tensorflow as tf

with tf.device('/cpu:0'):
    speed_test_1.speed_test_1()