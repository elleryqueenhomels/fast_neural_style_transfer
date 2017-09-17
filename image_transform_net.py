# Image Transformation Network - Deep Residual Convolutional Neural Network
# Used for Style Transferring


import tensorflow as tf


WEIGHT_INIT_STDDEV = 0.1


def conv2d(x, input_filters, output_filters, kernel_size, strides, mode='REFLECT'):
    with tf.variable_scope('conv2d') as scope:

        shape  = [kernel_size, kernel_size, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='weight')

        padding  = kernel_size // 2
        x_padded = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode=mode)

        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def conv2d_transpose(x, input_filters, output_filters, kernel_size, strides):
    with tf.variable_scope('conv2d_transpose') as scope:

        shape  = [kernel_size, kernel_size, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='weight')

        batch_size = tf.shape(x)[0]
        height     = tf.shape(x)[1] * strides
        width      = tf.shape(x)[2] * strides

        output_shape = [batch_size, height, width, output_filters]

        return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], name='conv_transpose')


def instance_norm(x):
    epsilon = 1e-3

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    x_normed  = tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    return x_normed


def residual(x, filters, kernel_size, strides):
    with tf.variable_scope('residual') as scope:

        conv1 = conv2d(x, filters, filters, kernel_size, strides)
        conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel_size, strides)

        return x + conv2


def transform(image):
    image = image / 127.5 - 1

    # mitigate border effects via padding a little before passing through
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope('conv1'):
        conv1 = tf.nn.relu(instance_norm(conv2d(image, 3, 32, 9, 1)))
    with tf.variable_scope('conv2'):
        conv2 = tf.nn.relu(instance_norm(conv2d(conv1, 32, 64, 3, 2))) # with downsampling
    with tf.variable_scope('conv3'):
        conv3 = tf.nn.relu(instance_norm(conv2d(conv2, 64, 128, 3, 2))) # with downsampling
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    with tf.variable_scope('deconv1'):
        deconv1 = tf.nn.relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2))) # with upsampling
    with tf.variable_scope('deconv2'):
        deconv2 = tf.nn.relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2))) # with upsampling
    with tf.variable_scope('convout'):
        convout = tf.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    output = (convout + 1) * 127.5

    # remove border effects via reducing padding
    height = tf.shape(output)[1]
    width  = tf.shape(output)[2]

    output = tf.slice(output, [0, 10, 10, 0], [-1, height - 20, width - 20, -1])

    return output

