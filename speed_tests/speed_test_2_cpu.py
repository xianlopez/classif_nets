import speed_test_2
import tensorflow as tf

with tf.device('/cpu:0'):
    speed_test_2.speed_test_2()