# coding=utf-8
# @author:xiaolin
# @file:ClassTensorFlow.py
# @time:2016/11/29 11:27

import tensorflow as tf
import numpy as np

x_data = np.random.rand(10)
y_data = x_data * 0.1 + 0.3

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = w * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step %20 ==0:
        print(step, sess.run(w), sess.run(b))