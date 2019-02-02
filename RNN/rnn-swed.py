#!/usr/bin/python3.6

import tensorflow as tf


vec = tf.constant([1, 2, 3, 4])
multiply = tf.constant([3])

matrix = tf.reshape(tf.tile(vec, multiply), [ multiply[0], tf.shape(vec)[0]])

print(matrix)
print("*")
with tf.Session() as sess:
    print(tf.Variable(sess.run(matrix)))
