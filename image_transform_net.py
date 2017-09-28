# Image Transformation Network - Deep Residual Convolutional Neural Network
# Used for Style Transferring

from __future__ import division

import tensorflow as tf


WEIGHT_INIT_STDDEV = 0.1


def conv2d(x, input_filters, output_filters, kernel_size, strides, relu=True, mode='REFLECT'):
    shape  = [kernel_size, kernel_size, input_filters, output_filters]
    weight = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='weight')

    padding  = kernel_size // 2
    x_padded = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode=mode)

    out = tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID')

    out = instance_norm(out, output_filters)

    if relu:
        out = tf.nn.relu(out)

    return out


def conv2d_transpose(x, input_filters, output_filters, kernel_size, strides):
    shape  = [kernel_size, kernel_size, output_filters, input_filters]
    weight = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='weight')

    batch_size = tf.shape(x)[0]
    height     = tf.shape(x)[1] * strides
    width      = tf.shape(x)[2] * strides

    output_shape = [batch_size, height, width, output_filters]

    out = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1])

    out = instance_norm(out, output_filters)

    out = tf.nn.relu(out)

    return out


def instance_norm(x, num_filters):
    epsilon = 1e-3

    shape = [num_filters]
    scale = tf.Variable(tf.ones(shape) , name='scale')
    shift = tf.Variable(tf.zeros(shape), name='shift')

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    x_normed  = tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    return scale * x_normed + shift


def residual(x, filters, kernel_size, strides):
    conv1 = conv2d(x, filters, filters, kernel_size, strides)
    conv2 = conv2d(conv1, filters, filters, kernel_size, strides, relu=False)

    return x + conv2


def transform(image):
    image = image / 127.5 - 1

    # mitigate border effects via padding a little before passing through
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope('conv1'):
        conv1 = conv2d(image, 3, 32, 9, 1)
    with tf.variable_scope('conv2'):
        conv2 = conv2d(conv1, 32, 64, 3, 2) # with downsampling
    with tf.variable_scope('conv3'):
        conv3 = conv2d(conv2, 64, 128, 3, 2) # with downsampling

    with tf.variable_scope('residual1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('residual2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('residual3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('residual4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('residual5'):
        res5 = residual(res4, 128, 3, 1)

    with tf.variable_scope('deconv1'):
        deconv1 = conv2d_transpose(res5, 128, 64, 3, 2) # with upsampling
    with tf.variable_scope('deconv2'):
        deconv2 = conv2d_transpose(deconv1, 64, 32, 3, 2) # with upsampling
    with tf.variable_scope('convout'):
        convout = tf.tanh(conv2d(deconv2, 32, 3, 9, 1, relu=False))

    output = (convout + 1) * 127.5

    # remove border effects via reducing padding
    height = tf.shape(output)[1]
    width  = tf.shape(output)[2]

    output = tf.slice(output, [0, 10, 10, 0], [-1, height - 20, width - 20, -1])

    return output

